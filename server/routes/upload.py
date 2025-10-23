import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..utils.env import get_settings
from ..utils.frames import FFmpegError, get_video_duration

router = APIRouter(prefix="/api", tags=["upload"])
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    settings = get_settings()
    if file.content_type not in {"video/mp4", "video/quicktime"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only MP4 and MOV videos are supported.",
        )

    extension = ".mp4" if file.content_type == "video/mp4" else ".mov"
    dest_name = f"{uuid.uuid4()}{extension}"
    dest_path = settings.upload_dir / dest_name

    size = 0
    try:
        with dest_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_video_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video exceeds {settings.max_video_mb} MB limit.",
                    )
                buffer.write(chunk)
    except HTTPException:
        if dest_path.exists():
            dest_path.unlink()
        raise
    except Exception as exc:  # pragma: no cover - unexpected errors
        if dest_path.exists():
            dest_path.unlink()
        logger.exception("Failed to save uploaded video")
        raise HTTPException(status_code=500, detail="Failed to store upload") from exc

    try:
        duration = get_video_duration(dest_path)
    except (FFmpegError, FileNotFoundError) as exc:
        if dest_path.exists():
            dest_path.unlink()
        logger.warning("Unable to inspect uploaded video: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid video file") from exc

    if duration > settings.max_video_seconds + 0.5:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video longer than {settings.max_video_seconds} seconds.",
        )

    public_url = f"/static/uploads/{dest_name}"
    return {"video_url": public_url, "duration_s": duration}
