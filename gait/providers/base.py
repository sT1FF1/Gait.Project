from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenParams:
    prompt: str
    negative_prompt: str = ""
    steps: int = 28
    guidance: float = 6.0
    width: int = 832
    height: int = 1216
    seed: Optional[int] = None
    duration: float = 5.0   # секунды (видео)
    fps: int = 16


class Provider(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def generate(self, params: GenParams) -> bytes:
        """Возвращает PNG-байты."""

    def generate_video(self, params: GenParams) -> tuple[bytes, str]:
        """Возвращает (байты, расширение файла)."""
        raise NotImplementedError("Этот бэкенд не поддерживает видео")
