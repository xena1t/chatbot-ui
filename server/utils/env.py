import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    vllm_host: str
    vllm_port: int
    model_name: str
    data_root: Path
    static_root: Path
    upload_dir: Path
    frames_dir: Path
    cors_origin: str
    max_video_seconds: int
    max_video_mb: int
    frame_count: int
    frame_max_w: int
    frame_max_h: int

    @property
    def max_video_bytes(self) -> int:
        return self.max_video_mb * 1024 * 1024


DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def _resolve_data_root() -> Path:
    candidates = []
    persistent = os.getenv("RUNPOD_PERSISTENT_DIR")
    if persistent:
        candidates.append(Path(persistent))
    candidates.append(Path("/workspace"))
    candidates.append(Path("./data"))

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        if path.exists() and os.access(path, os.W_OK):
            return path.resolve()

    # Final fallback to current directory data folder.
    fallback = Path("./data")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    vllm_host = os.getenv("VLLM_HOST", "127.0.0.1")
    vllm_port = int(os.getenv("VLLM_PORT", "8001"))
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)

    data_root = _resolve_data_root()
    static_root = data_root / "static"
    upload_dir = static_root / "uploads"
    frames_dir = static_root / "frames"

    for directory in (static_root, upload_dir, frames_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cors_origin = os.getenv("CORS_ORIGIN", "*")
    max_video_seconds = int(os.getenv("MAX_VIDEO_SECONDS", "30"))
    max_video_mb = int(os.getenv("MAX_VIDEO_MB", "50"))
    frame_count = int(os.getenv("FRAME_COUNT", "8"))
    frame_max_w = int(os.getenv("FRAME_MAX_W", "1280"))
    frame_max_h = int(os.getenv("FRAME_MAX_H", "720"))

    return Settings(
        host=host,
        port=port,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        model_name=model_name,
        data_root=data_root,
        static_root=static_root,
        upload_dir=upload_dir,
        frames_dir=frames_dir,
        cors_origin=cors_origin,
        max_video_seconds=max_video_seconds,
        max_video_mb=max_video_mb,
        frame_count=frame_count,
        frame_max_w=frame_max_w,
        frame_max_h=frame_max_h,
    )


__all__ = ["get_settings", "Settings"]
