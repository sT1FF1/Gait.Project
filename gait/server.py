import json
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .guard import check_prompt
from .providers import BACKENDS, GenParams

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

app = FastAPI(title="Gait image generator")
_cfg = yaml.safe_load((ROOT / "models.yaml").read_text())["models"]
_models = {m["id"]: m for m in _cfg}
_providers = {}


class GenRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    steps: int | None = None
    guidance: float | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None


@app.get("/api/models")
def models():
    return [{"id": m["id"], "name": m["name"], "kind": m.get("kind", "image"), "defaults": m.get("defaults", {})}
            for m in _cfg]


@app.get("/api/catalog")
def catalog():
    return json.loads((ROOT / "data" / "catalog.json").read_text())


class VideoRequest(GenRequest):
    duration: float = Field(default=5, ge=5, le=30)


def _prepare(req: GenRequest, kind: str):
    cfg = _models.get(req.model)
    if not cfg or cfg.get("kind", "image") != kind:
        raise HTTPException(404, "Неизвестная модель")
    try:
        check_prompt(req.prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.model not in _providers:
        _providers[req.model] = BACKENDS[cfg["backend"]](cfg)
    d = cfg.get("defaults", {})
    prompt = " ".join(x for x in (d.get("prompt_prefix", ""), req.prompt, d.get("prompt_suffix", "")) if x)
    negative = ", ".join(x for x in (d.get("negative_prompt", ""), req.negative_prompt) if x)
    params = GenParams(
        prompt=prompt,
        negative_prompt=negative,
        steps=req.steps or d.get("steps", 28),
        guidance=req.guidance or d.get("guidance", 6.0),
        width=req.width or d.get("width", 832),
        height=req.height or d.get("height", 1216),
        seed=req.seed,
        fps=d.get("fps", 16),
    )
    return _providers[req.model], params


def _save(data: bytes, ext: str) -> str:
    name = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.{ext}"
    (OUT / name).write_bytes(data)
    return f"/outputs/{name}"


@app.post("/api/generate")
def generate(req: GenRequest):
    provider, params = _prepare(req, "image")
    try:
        png = provider.generate(params)
    except Exception as e:
        raise HTTPException(500, f"Ошибка генерации: {e}")
    return {"url": _save(png, "png"), "kind": "image"}


@app.post("/api/generate-video")
def generate_video(req: VideoRequest):
    provider, params = _prepare(req, "video")
    params.duration = req.duration
    try:
        data, ext = provider.generate_video(params)
    except Exception as e:
        raise HTTPException(500, f"Ошибка генерации: {e}")
    return {"url": _save(data, ext), "kind": "video", "ext": ext}


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
