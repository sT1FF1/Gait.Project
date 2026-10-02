import json
import random
import time
import uuid
from pathlib import Path

import httpx

from .base import GenParams, Provider

ROOT = Path(__file__).resolve().parent.parent.parent


def _default_workflow(ckpt: str) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{negative_prompt}}", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": "{{width}}", "height": "{{height}}", "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": "{{seed}}", "steps": "{{steps}}", "cfg": "{{guidance}}",
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "gait"}},
    }


def _fill(node, values: dict):
    """Подставляет {{name}} в значения графа; строка из одного плейсхолдера сохраняет тип."""
    if isinstance(node, dict):
        return {k: _fill(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, values) for v in node]
    if isinstance(node, str) and "{{" in node:
        stripped = node.strip()
        for k, v in values.items():
            if stripped == "{{%s}}" % k:
                return v
        for k, v in values.items():
            node = node.replace("{{%s}}" % k, str(v))
    return node


class ComfyUIProvider(Provider):
    """Клиент к ComfyUI: POST /prompt, опрос /history, загрузка /view.

    Либо `checkpoint` (встроенный txt2img-граф), либо `workflow_file` — свой граф в API-формате
    с плейсхолдерами {{prompt}}, {{negative_prompt}}, {{width}}, {{height}}, {{steps}}, {{guidance}}, {{seed}}, а для видео ещё {{frames}}, {{fps}}, {{duration}}.
    """

    def _workflow(self) -> dict:
        if self.cfg.get("workflow_file"):
            return json.loads((ROOT / self.cfg["workflow_file"]).read_text())
        return _default_workflow(self.cfg["checkpoint"])

    def _run(self, p: GenParams) -> tuple[bytes, str]:
        base = self.cfg["url"].rstrip("/")
        seed = p.seed if p.seed is not None else random.randint(0, 2**32 - 1)
        graph = _fill(self._workflow(), {
            "prompt": p.prompt, "negative_prompt": p.negative_prompt, "width": p.width,
            "height": p.height, "steps": p.steps, "guidance": p.guidance, "seed": seed,
            "fps": p.fps, "duration": p.duration, "frames": int(round(p.duration * p.fps)) + 1,
        })
        timeout = self.cfg.get("timeout", 600)
        with httpx.Client(timeout=30) as c:
            r = c.post(f"{base}/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex})
            if r.status_code != 200:
                raise RuntimeError(f"ComfyUI отклонил граф: {r.text[:500]}")
            pid = r.json()["prompt_id"]
            deadline = time.time() + timeout
            while time.time() < deadline:
                hist = c.get(f"{base}/history/{pid}").json().get(pid)
                if hist and hist.get("outputs"):
                    for out in hist["outputs"].values():
                        # SaveImage -> images; VideoHelperSuite/SaveVideo -> gifs / videos / images (animated)
                        for key in ("images", "gifs", "videos"):
                            for f in out.get(key, []):
                                v = c.get(f"{base}/view", params={
                                    "filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")}, timeout=120)
                                v.raise_for_status()
                                return v.content, Path(f["filename"]).suffix.lstrip(".").lower() or "png"
                    raise RuntimeError("ComfyUI не вернул файлов")
                time.sleep(1)
        raise TimeoutError("ComfyUI: превышено время ожидания")

    def generate(self, p: GenParams) -> bytes:
        return self._run(p)[0]

    def generate_video(self, p: GenParams) -> tuple[bytes, str]:
        if not self.cfg.get("workflow_file"):
            raise RuntimeError("Для видео нужен workflow_file (граф Wan / AnimateDiff и т.п.)")
        return self._run(p)
