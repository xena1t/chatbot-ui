# RunPod Qwen3-VL Chat UI

An end-to-end demo for chatting with **Qwen3-VL-32B-Instruct** on RunPod. The
system loads Hugging Face models with 🤗 Transformers inside the FastAPI backend,
streams responses to the browser via Server-Sent Events, and supports uploading a
short video whose frames are extracted and sent to the model.

## Features

- ✅ Single-container deployment for RunPod GPU pods
- ✅ FastAPI backend with `/api/upload` and `/api/chat/stream`
- ✅ Automatic frame extraction (ffmpeg) with configurable limits
- ✅ SSE token streaming to a React + Vite frontend
- ✅ Assets, uploads, and frames stored on the RunPod persistent volume

## Repository layout

```
.
├── README.md
├── .env.example
├── server/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   ├── run_all.sh
│   ├── uvicorn_logging.yaml
│   ├── adapters/
│   │   └── qwen.py
│   ├── routes/
│   │   ├── chat.py
│   │   └── upload.py
│   ├── utils/
│   │   ├── env.py
│   │   └── frames.py
│   ├── static/              # mount point for uploads & extracted frames
│   └── web/                 # compiled frontend assets served by FastAPI
└── webapp/                  # React + Vite source
    ├── index.html
    ├── package.json
    ├── src/
    │   ├── App.tsx
    │   ├── lib/sse.ts
    │   ├── main.tsx
    │   └── styles.css
    └── vite.config.ts
```

## Environment variables

Copy `.env.example` into `.env` (or configure via RunPod UI). Important variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | FastAPI bind host | `0.0.0.0` |
| `PORT` | FastAPI bind port | `8000` |
| `MODEL_NAME` | Hugging Face identifier for Qwen | `Qwen/Qwen3-VL-32B-Instruct` |
| `TRANSFORMERS_DEVICE` | Target device (`auto`, `cuda`, or `cpu`) | `auto` |
| `CORS_ORIGIN` | Comma-separated list of allowed origins (`*` for dev) | `*` |
| `MAX_VIDEO_SECONDS` | Maximum upload duration | `30` |
| `MAX_VIDEO_MB` | Maximum upload size | `50` |
| `FRAME_COUNT` | Frames sampled per video | `8` |
| `FRAME_MAX_W` / `FRAME_MAX_H` | Max resolution for frames | `1280x720` |

`RUNPOD_PERSISTENT_DIR` is automatically discovered; uploads and frames live in
`$DATA_ROOT/static/{uploads,frames}`. If the variable is missing, `/workspace`
(and finally `./data`) is used instead.

## Building the container image

The provided Dockerfile creates a single RunPod-ready image with the FastAPI app
and bundled Transformers backend. Build it from the repository root:

```bash
docker build -t runpod-qwen3vl -f server/Dockerfile .
```

This multi-stage build installs the frontend dependencies, compiles the Vite app
into `server/web/`, installs the Python requirements (including `transformers`
and `torch`), and sets the entrypoint to `run_all.sh`.

## Running locally (GPU required)

1. Install [ffmpeg](https://ffmpeg.org/), Python 3.10+, and Node.js 18+.
2. Install server dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```
3. Build the frontend assets:
   ```bash
   cd webapp
   npm install
   npm run build
   ```
4. Launch the stack:
   ```bash
   cd ..
   HOST=0.0.0.0 PORT=8000 bash server/run_all.sh
   ```

The script boots Uvicorn on `0.0.0.0:8000`, loads the configured Transformers
model, and serves the React application alongside `/api/upload`,
`/api/chat/stream`, `/healthz`, and `/static/*`.

## API overview

- `POST /api/upload`
  - Accepts `multipart/form-data` with a single `file` field (MP4/MOV).
  - Validates size & duration, stores the video as `/static/uploads/<uuid>.mp4`.
  - Responds with `{ "video_url": "/static/uploads/<uuid>.mp4", "duration_s": 12.3 }`.
- `POST /api/chat/stream`
  - Body: `{ messages: [{role, content}, ...], params?: {...}, video_url?: "..." }`.
  - Extracts ~8 uniformly sampled frames, publishes them as `/static/frames/<id>/frame_xx.jpg`.
  - Streams SSE events: `frames` (array of public URLs), `token` (delta text),
    optional `error`, then `data: [DONE]` when complete.

## Frontend workflow

1. User types a prompt and optionally selects a video (MP4/MOV ≤ 30s).
2. The video is uploaded to `/api/upload`.
3. The chat history and optional `video_url` are posted to `/api/chat/stream`.
4. The client consumes the SSE stream, updates the assistant message in real-time,
   and shows the extracted frames underneath the user message.

## Hardening tips

- Change `CORS_ORIGIN` to the actual domain in production.
- Configure HTTPS termination at the RunPod proxy level.
- Add authentication (e.g., RunPod secret) if exposing the endpoint publicly.
- Periodically clean the `$DATA_ROOT/static` directory if storage is limited.

## Health check

`GET /healthz` responds with `{ "status": "ok" }` and can be used by RunPod to
monitor the container.

## License

MIT
