"""FastAPI server: serves the dashboard, MJPEG video, WebSocket feed, controls."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from .config import Config, load_config
from .pipeline import Pipeline

log = logging.getLogger("life.server")

_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.svg"
)

pipeline: Optional[Pipeline] = None
app = FastAPI(title="LIFE")


def create_app(config: Optional[Config] = None) -> FastAPI:
    global pipeline
    config = config or load_config()
    pipeline = Pipeline(config)

    @app.on_event("startup")
    def _startup() -> None:
        pipeline.start()
        log.info("LIFE pipeline started.")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        if pipeline:
            pipeline.stop()

    return app


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = os.path.join(_WEB_DIR, "index.html")
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/logo.svg")
def logo() -> Response:
    if os.path.exists(_LOGO_PATH):
        return FileResponse(_LOGO_PATH, media_type="image/svg+xml")
    return Response(status_code=404)


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = pipeline.state.get_frame() if pipeline else None
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(1 / 25.0)


@app.get("/video")
def video() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def status() -> JSONResponse:
    if not pipeline:
        return JSONResponse({"running": False})
    return JSONResponse(pipeline.state.snapshot_status())


@app.get("/api/events")
def events(limit: int = 50) -> JSONResponse:
    if not pipeline:
        return JSONResponse([])
    return JSONResponse(pipeline.state.recent_events(limit))


@app.post("/api/ack")
def ack() -> JSONResponse:
    if pipeline:
        pipeline.acknowledge_alarm()
    return JSONResponse({"ok": True})


@app.post("/api/detector/{name}")
def toggle_detector(name: str, enabled: bool = True) -> JSONResponse:
    ok = pipeline.toggle_detector(name, enabled) if pipeline else False
    return JSONResponse({"ok": ok, "detector": name, "enabled": enabled})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    last_event_id = 0
    try:
        while True:
            if pipeline:
                payload = {
                    "status": pipeline.state.snapshot_status(),
                    "events": [
                        e for e in pipeline.state.recent_events(20)
                        if e.get("id", 0) > last_event_id
                    ],
                }
                if payload["events"]:
                    last_event_id = max(e["id"] for e in payload["events"])
                await websocket.send_json(payload)
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover
        log.debug("WebSocket closed: %s", exc)
