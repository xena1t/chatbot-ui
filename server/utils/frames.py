import shutil
import subprocess
from pathlib import Path
from typing import List


class FFmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


def get_video_duration(path: Path) -> float:
    """Return the duration of a video in seconds using ffprobe."""
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(exc.stderr.strip()) from exc

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise FFmpegError(f"Unable to parse duration for {path}") from exc

    return max(duration, 0.0)


def extract_uniform_frames(
    video_path: Path,
    output_dir: Path,
    frame_count: int,
    max_width: int,
    max_height: int,
) -> List[Path]:
    """Extract approximately frame_count uniformly spaced frames."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = max(get_video_duration(video_path), 0.001)
    fps = max(frame_count / duration, 1.0 / duration)

    scale_filter = (
        f"scale='min({max_width},iw)':'min({max_height},ih)':"
        "force_original_aspect_ratio=decrease"
    )
    filters = f"fps={fps},{scale_filter}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filters,
        "-frames:v",
        str(frame_count),
        "-q:v",
        "2",
        str(output_dir / "frame_%02d.jpg"),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(exc.stderr.strip() if exc.stderr else "ffmpeg failed") from exc

    frames = sorted(output_dir.glob("frame_*.jpg"))
    return frames


__all__ = ["extract_uniform_frames", "get_video_duration", "FFmpegError"]
