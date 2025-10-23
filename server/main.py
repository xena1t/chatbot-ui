from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .routes import chat, upload
from .utils.env import get_settings

settings = get_settings()

app = FastAPI(title="RunPod Qwen3-VL Chat", version="0.1.0")

allow_origins: List[str]
if settings.cors_origin == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in settings.cors_origin.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(chat.router)

app.mount("/static", StaticFiles(directory=settings.static_root, check_dir=True), name="static")

web_dir = Path(__file__).resolve().parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return JSONResponse({"status": "ok"})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
