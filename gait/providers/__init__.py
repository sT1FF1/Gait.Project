from .base import Provider, GenParams
from .diffusers_local import DiffusersProvider
from .a1111 import A1111Provider
from .comfyui import ComfyUIProvider

BACKENDS = {"diffusers": DiffusersProvider, "a1111": A1111Provider, "comfyui": ComfyUIProvider}

__all__ = ["Provider", "GenParams", "BACKENDS"]
