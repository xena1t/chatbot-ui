# syntax=docker/dockerfile:1.5

FROM node:18-bullseye AS web-builder
WORKDIR /app
COPY webapp/package*.json ./webapp/
RUN cd webapp && npm install
COPY webapp ./webapp
RUN cd webapp && npm run build

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

COPY server /app/server
COPY --from=web-builder /app/server/web /app/server/web

RUN chmod +x /app/server/run_all.sh

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /app/server/requirements.txt

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_NAME=Qwen/Qwen3-VL-32B-Instruct \
    TRANSFORMERS_DEVICE=auto

EXPOSE 8000

CMD ["bash", "/app/server/run_all.sh"]
