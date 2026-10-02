"""Жёсткие ограничения, не зависящие от выбранной модели.

Контент для взрослых допустим, но запросы с несовершеннолетними блокируются всегда.
"""
import re

_MINOR = re.compile(
    r"\b(child|children|kid|kids|minor|underage|under-age|loli|lolita|shota|toddler|"
    r"infant|baby|preteen|pre-teen|schoolgirl|schoolboy|teen(?:ager)?|[1-9]\s*(?:yo|y\.o\.|years?\s*old)|1[0-7]\s*(?:yo|y\.o\.|years?\s*old))\b",
    re.IGNORECASE,
)


def check_prompt(prompt: str) -> None:
    if _MINOR.search(prompt):
        raise ValueError("Запрос заблокирован: упоминание несовершеннолетних недопустимо.")
