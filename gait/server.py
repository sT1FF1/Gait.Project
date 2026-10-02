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
    return [{"id": m["id"], "name": m["name"], "defaults": m.get("defaults", {})} for m in _cfg]


@app.post("/api/generate")
def generate(req: GenRequest):
    cfg = _models.get(req.model)
    if not cfg:
        raise HTTPException(404, "Неизвестная модель")
    try:
        check_prompt(req.prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.model not in _providers:
        _providers[req.model] = BACKENDS[cfg["backend"]](cfg)
    d = cfg.get("defaults", {})
    params = GenParams(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        steps=req.steps or d.get("steps", 28),
        guidance=req.guidance or d.get("guidance", 6.0),
        width=req.width or d.get("width", 832),
        height=req.height or d.get("height", 1216),
        seed=req.seed,
    )
    try:
        png = _providers[req.model].generate(params)
    except Exception as e:
        raise HTTPException(500, f"Ошибка генерации: {e}")
    name = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.png"
    (OUT / name).write_bytes(png)
    return {"url": f"/outputs/{name}"}


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
