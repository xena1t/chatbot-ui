import asyncio
import json
import logging
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..adapters.qwen import stream_chat_completion
from ..utils.env import get_settings
from ..utils.frames import FFmpegError, extract_uniform_frames

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    params: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None
    model: Optional[str] = None


def _resolve_static_path(static_url: str) -> Path:
    settings = get_settings()
    parsed = urlparse(static_url)
    path = Path(parsed.path)
    if not str(path).startswith("/static/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid video URL")
    relative = path.as_posix()[len("/static/"):]
    full_path = (settings.static_root / relative).resolve()
    try:
        full_path.relative_to(settings.static_root)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid video URL") from exc
    if not full_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return full_path


def _format_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _prepare_frames(video_url: str) -> Tuple[List[str], List[Path]]:
    settings = get_settings()
    video_path = _resolve_static_path(video_url)

    frame_folder = settings.frames_dir / str(uuid.uuid4())
    try:
        frames = extract_uniform_frames(
            video_path=video_path,
            output_dir=frame_folder,
            frame_count=settings.frame_count,
            max_width=settings.frame_max_w,
            max_height=settings.frame_max_h,
        )
    except (FFmpegError, FileNotFoundError) as exc:
        logger.error("Frame extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to process video frames") from exc

    public_urls: List[str] = []
    if not frames:
        raise HTTPException(status_code=500, detail="No frames extracted")

    for frame in frames:
        relative = frame.relative_to(settings.static_root)
        public_path = f"/static/{relative.as_posix()}"
        public_urls.append(public_path)

    return public_urls, frames


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    messages = [message.dict() for message in payload.messages]

    async def event_generator():
        queue: asyncio.Queue[str] = asyncio.Queue()
        first_token = asyncio.Event()

        async def heartbeat() -> None:
            try:
                # Send an initial comment immediately to keep the connection open.
                await queue.put(":\n\n")
                while not first_token.is_set():
                    await asyncio.sleep(5)
                    if first_token.is_set():
                        break
                    await queue.put(":\n\n")
            except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
                pass

        async def produce_events() -> None:
            try:
                frame_urls: List[str] = []
                frame_paths: List[Path] = []
                if payload.video_url:
                    frame_urls, frame_paths = await asyncio.to_thread(
                        _prepare_frames, payload.video_url
                    )

                if frame_urls:
                    await queue.put(_format_sse({"event": "frames", "frames": frame_urls}))

                async for event in stream_chat_completion(
                    messages=messages,
                    params=payload.params,
                    image_paths=frame_paths or None,
                    model=payload.model,
                ):
                    if await request.is_disconnected():
                        logger.info("Client disconnected from stream")
                        break
                    if event.get("event") == "token":
                        first_token.set()
                    await queue.put(_format_sse(event))
            except HTTPException as exc:
                logger.warning("Frame preparation failed: %s", exc.detail)
                first_token.set()
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                await queue.put(_format_sse({"event": "error", "message": detail}))
            except Exception as exc:  # pragma: no cover - unexpected
                logger.exception("Unexpected error during streaming")
                await queue.put(_format_sse({"event": "error", "message": str(exc)}))
            finally:
                first_token.set()
                await queue.put("data: [DONE]\n\n")

        heartbeat_task = asyncio.create_task(heartbeat())
        producer_task = asyncio.create_task(produce_events())

        try:
            while True:
                item = await queue.get()
                yield item
                if item == "data: [DONE]\n\n":
                    break
        finally:
            heartbeat_task.cancel()
            producer_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            with suppress(asyncio.CancelledError):
                await producer_task

    return StreamingResponse(event_generator(), media_type="text/event-stream")
