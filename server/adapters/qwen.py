import json
import logging
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

import httpx

from ..utils.env import get_settings

logger = logging.getLogger(__name__)


def _ensure_content_is_list(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _merge_images_into_messages(
    messages: List[Dict[str, Any]],
    image_urls: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    prepared = []
    for item in messages:
        prepared.append(
            {
                "role": item.get("role", "user"),
                "content": _ensure_content_is_list(item.get("content")),
            }
        )

    image_urls = list(image_urls or [])
    if not image_urls:
        return prepared

    for message in reversed(prepared):
        if message["role"] == "user":
            content = message.setdefault("content", [])
            if not isinstance(content, list):
                content = _ensure_content_is_list(content)
                message["content"] = content
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
                for url in image_urls
            )
            break
    else:
        prepared.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    *(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                        for url in image_urls
                    ),
                ],
            }
        )

    return prepared


async def stream_chat_completion(
    messages: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
    image_urls: Optional[Iterable[str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    settings = get_settings()
    request_payload: Dict[str, Any] = {
        "model": settings.model_name,
        "stream": True,
        "messages": _merge_images_into_messages(messages, image_urls),
    }
    if params:
        request_payload.update(params)

    url = f"http://{settings.vllm_host}:{settings.vllm_port}/v1/chat/completions"
    timeout = httpx.Timeout(300.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=request_payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    continue

                if not data:
                    continue
                if data == "[DONE]":
                    yield {"event": "done"}
                    break

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Unable to decode payload from vLLM: %s", data)
                    continue

                choices = payload.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    if "content" in delta:
                        yield {
                            "event": "token",
                            "text": delta["content"],
                        }
                finish_reason = None
                if choices:
                    finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    yield {"event": "finish", "reason": finish_reason}


__all__ = ["stream_chat_completion"]
