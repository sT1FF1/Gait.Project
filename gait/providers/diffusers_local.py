import io
import threading

from .base import GenParams, Provider


class DiffusersProvider(Provider):
    """Локальная генерация через diffusers. Пайплайн грузится лениво при первом запросе."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self):
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        cls = StableDiffusionXLPipeline if self.cfg.get("pipeline", "sdxl") == "sdxl" else StableDiffusionPipeline
        path = self.cfg["path"]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if path.endswith((".safetensors", ".ckpt")):
            pipe = cls.from_single_file(path, torch_dtype=dtype)
        else:
            pipe = cls.from_pretrained(path, torch_dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipe.to(device)
        # Фильтр безопасности на уровне пайплайна отключён: решение об уровне контента — на стороне оператора
        if hasattr(self._pipe, "safety_checker"):
            self._pipe.safety_checker = None

    def generate(self, p: GenParams) -> bytes:
        import torch

        with self._lock:  # один запрос на GPU за раз
            if self._pipe is None:
                self._load()
            gen = None
            if p.seed is not None:
                gen = torch.Generator(device=self._pipe.device).manual_seed(p.seed)
            img = self._pipe(
                prompt=p.prompt,
                negative_prompt=p.negative_prompt or None,
                num_inference_steps=p.steps,
                guidance_scale=p.guidance,
                width=p.width,
                height=p.height,
                generator=gen,
            ).images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
