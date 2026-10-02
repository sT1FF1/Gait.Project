import base64

import httpx

from .base import GenParams, Provider


class A1111Provider(Provider):
    """Клиент к AUTOMATIC1111 / Forge (--api): POST /sdapi/v1/txt2img."""

    def generate(self, p: GenParams) -> bytes:
        payload = {
            "prompt": p.prompt,
            "negative_prompt": p.negative_prompt,
            "steps": p.steps,
            "cfg_scale": p.guidance,
            "width": p.width,
            "height": p.height,
            "seed": p.seed if p.seed is not None else -1,
        }
        if self.cfg.get("checkpoint"):
            payload["override_settings"] = {"sd_model_checkpoint": self.cfg["checkpoint"]}
        r = httpx.post(self.cfg["url"].rstrip("/") + "/sdapi/v1/txt2img", json=payload, timeout=600)
        r.raise_for_status()
        return base64.b64decode(r.json()["images"][0])
