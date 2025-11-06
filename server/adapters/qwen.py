import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Sequence

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
)

try:  # transformers<4.46 does not expose this helper
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - optional depending on version
    AutoModelForImageTextToText = None  # type: ignore[assignment]
from transformers.modeling_utils import PreTrainedModel

from ..utils.env import get_settings

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - pillow should be installed
    raise RuntimeError("Pillow is required for vision-language models") from exc

logger = logging.getLogger(__name__)


_VISION_LANGUAGE_KEYWORDS: Sequence[str] = (
    "vl",
    "vision",
    "video",
    "vila",
    "onevision",
    "internvl",
    "llava",
)


@dataclass
class _ModelBundle:
    processor: Any
    tokenizer: Any
    model: PreTrainedModel
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]
    uses_vision_language: bool


_MODEL_CACHE: Dict[str, _ModelBundle] = {}
_CACHE_LOCK = asyncio.Lock()
_MODEL_LOCKS: Dict[str, asyncio.Lock] = {}


def _ensure_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("type") == "image_url" and item.get("image_url"):
                    url = item["image_url"].get("url")
                    if url:
                        parts.append(f"[image: {url}]")
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _format_prompt(
    messages: List[Dict[str, Any]], image_refs: Optional[Iterable[str]]
) -> str:
    segments: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        role_label = "User" if role == "user" else "Assistant"
        text = _ensure_text(message.get("content"))
        segments.append(f"{role_label}: {text}" if text else f"{role_label}:")

    refs = list(image_refs or [])
    if refs:
        segments.append("User shared the following image frames:")
        segments.extend(refs)

    segments.append("Assistant:")
    return "\n".join(segments).strip() + "\n"


def _build_conversation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conversation: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        text = _ensure_text(message.get("content"))
        content: List[Dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        if role == "system" and not content:
            continue
        conversation.append({"role": role, "content": content})
    return conversation

def _load_images(frame_paths: Sequence[Path]) -> List[Any]:
    images: List[Any] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            images.append(image.convert("RGB"))
    return images

def _load_images(frame_paths: Sequence[Path]) -> List[Any]:
    images: List[Any] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            images.append(image.convert("RGB"))
    return images


def _prepare_multimodal_inputs(
    processor: Any,
    messages: List[Dict[str, Any]],
    frame_paths: Sequence[Path],
) -> Dict[str, Any]:
    conversation = _build_conversation(messages)
    images = _load_images(frame_paths)

    if images:
        if not conversation or conversation[-1]["role"] != "user":
            conversation.append({"role": "user", "content": []})
        existing_content = conversation[-1].setdefault("content", [])
        image_items = [{"type": "image", "image": image} for image in images]
        conversation[-1]["content"] = image_items + existing_content

    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: Dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        return processor(**processor_kwargs)

    logger.warning("Processor missing chat template; falling back to text-only encoding")
    prompt = _format_prompt(
        messages,
        [path.as_posix() for path in frame_paths],
    )
    return processor(prompt, return_tensors="pt", add_special_tokens=True)


async def _load_model(model_name: str) -> _ModelBundle:
    async with _CACHE_LOCK:
        bundle = _MODEL_CACHE.get(model_name)
        if bundle:
            return bundle
        lock = _MODEL_LOCKS.setdefault(model_name, asyncio.Lock())

    async with lock:
        async with _CACHE_LOCK:
            bundle = _MODEL_CACHE.get(model_name)
            if bundle:
                return bundle

        settings = get_settings()

        def _load() -> _ModelBundle:
            load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            target_device = settings.transformers_device
            if target_device == "auto":
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
            else:
                if target_device.startswith("cuda"):
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["torch_dtype"] = torch.float32
            lower_name = model_name.lower()
            uses_vision_language = any(
                keyword in lower_name for keyword in _VISION_LANGUAGE_KEYWORDS
            )

            if uses_vision_language:
                processor = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )

                load_attempts: List[str] = []

                vision_loaders: List[Any] = []
                if AutoModelForImageTextToText is not None:
                    vision_loaders.append(AutoModelForImageTextToText)
                vision_loaders.append(AutoModelForVision2Seq)
                vision_loaders.append(AutoModelForCausalLM)
                vision_loaders.append(AutoModel)

                model = None
                for loader in vision_loaders:
                    try:
                        model = loader.from_pretrained(model_name, **load_kwargs)
                        break
                    except ValueError as err:
                        load_attempts.append(f"{loader.__name__}: {err}")
                    except Exception as err:  # pragma: no cover - defensive
                        load_attempts.append(f"{loader.__name__}: {err}")

                if model is None:
                    error_details = " | ".join(load_attempts)
                    raise RuntimeError(
                        "Unable to load vision-language model '%s': %s"
                        % (model_name, error_details or "no loaders succeeded")
                    )

                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, trust_remote_code=True
                    )
            else:
                processor = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                tokenizer = processor
            if target_device != "auto":
                model.to(target_device)

            try:
                reference_parameter = next(model.parameters())
            except StopIteration:  # pragma: no cover - defensive, models always have params
                reference_parameter = None

            bundle_device: Optional[torch.device]
            bundle_dtype: Optional[torch.dtype]
            if target_device != "auto":
                bundle_device = torch.device(target_device)
                bundle_dtype = (
                    reference_parameter.dtype if reference_parameter is not None else None
                )
            else:
                if reference_parameter is not None:
                    bundle_device = reference_parameter.device
                    bundle_dtype = reference_parameter.dtype
                else:
                    bundle_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                    bundle_dtype = torch.float16 if bundle_device.type == "cuda" else torch.float32
            model.eval()
            return _ModelBundle(
                processor=processor,
                tokenizer=tokenizer,
                model=model,
                device=bundle_device,
                dtype=bundle_dtype,
                uses_vision_language=uses_vision_language,
            )

        bundle = await asyncio.to_thread(_load)
        async with _CACHE_LOCK:
            _MODEL_CACHE[model_name] = bundle
        return bundle


def _build_generate_kwargs(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    max_new_tokens = 512
    temperature = 0.7
    top_p = None
    top_k = None
    repetition_penalty = None

    if params:
        if "max_new_tokens" in params:
            try:
                max_new_tokens = int(params["max_new_tokens"])
            except (TypeError, ValueError):
                logger.warning("Invalid max_new_tokens value: %s", params["max_new_tokens"])
        if "temperature" in params:
            try:
                temperature = float(params["temperature"])
            except (TypeError, ValueError):
                logger.warning("Invalid temperature value: %s", params["temperature"])
        if "top_p" in params:
            try:
                top_p = float(params["top_p"])
            except (TypeError, ValueError):
                logger.warning("Invalid top_p value: %s", params["top_p"])
        if "top_k" in params:
            try:
                top_k = int(params["top_k"])
            except (TypeError, ValueError):
                logger.warning("Invalid top_k value: %s", params["top_k"])
        if "repetition_penalty" in params:
            try:
                repetition_penalty = float(params["repetition_penalty"])
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid repetition_penalty value: %s", params["repetition_penalty"]
                )

    max_new_tokens = max(1, min(max_new_tokens, 2048))
    kwargs["max_new_tokens"] = max_new_tokens

    do_sample = temperature is not None and temperature > 0
    kwargs["do_sample"] = do_sample
    if do_sample:
        kwargs["temperature"] = max(0.01, temperature)
        if top_p is not None and 0 < top_p <= 1:
            kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            kwargs["top_k"] = top_k
    if repetition_penalty is not None and repetition_penalty > 0:
        kwargs["repetition_penalty"] = repetition_penalty

    return kwargs


async def stream_chat_completion(
    messages: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
    image_paths: Optional[Iterable[Path]] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    settings = get_settings()
    model_name = model or settings.model_name

    bundle = await _load_model(model_name)

    processor = bundle.processor

    def _encode() -> Dict[str, Any]:
        prompt = _format_prompt(
            messages,
            [path.as_posix() for path in image_paths] if image_paths else None,
        )
        if bundle.uses_vision_language:
            if image_paths:
                return _prepare_multimodal_inputs(
                    processor,
                    messages,
                    list(image_paths),
                )
            tokenizer_to_use = getattr(bundle.tokenizer, "__call__", None)
            if callable(tokenizer_to_use):
                return bundle.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
        return processor(
            text=prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

    encoded = await asyncio.to_thread(_encode)

    tokenizer = bundle.tokenizer

    try:
        model_device = bundle.device or next(bundle.model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive
        model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_dtype = bundle.dtype
    if model_dtype is None:
        try:
            model_dtype = next(bundle.model.parameters()).dtype
        except StopIteration:  # pragma: no cover - defensive
            model_dtype = torch.float16 if model_device.type == "cuda" else torch.float32

    moved_inputs: Dict[str, Any] = {}
    for key, value in encoded.items():
        if hasattr(value, "to"):
            to_kwargs: Dict[str, Any] = {"device": model_device}
            if isinstance(model_device, torch.device) and model_device.type == "cuda":
                to_kwargs["non_blocking"] = True
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                to_kwargs["dtype"] = model_dtype
            moved_inputs[key] = value.to(**to_kwargs)
        else:
            moved_inputs[key] = value
    encoded = moved_inputs

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = _build_generate_kwargs(params)
    generate_kwargs["streamer"] = streamer

    for key, value in encoded.items():
        generate_kwargs[key] = value

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def _forward_stream() -> None:
        for text in streamer:
            loop.call_soon_threadsafe(queue.put_nowait, text)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    def _generate() -> None:
        with torch.inference_mode():
            bundle.model.generate(**generate_kwargs)

    forward_thread = threading.Thread(target=_forward_stream, daemon=True)
    forward_thread.start()

    generation_future = loop.run_in_executor(None, _generate)

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if chunk:
                yield {"event": "token", "text": chunk}
        await generation_future
        yield {"event": "finish", "reason": "stop"}
    finally:
        if forward_thread.is_alive():
            forward_thread.join(timeout=0.1)


__all__ = ["stream_chat_completion"]
