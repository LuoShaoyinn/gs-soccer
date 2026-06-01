import os
from pathlib import Path

import cairosvg

_TEXTURES_DIR = Path(__file__).resolve().parent.parent / "assets" / "textures"
_CACHE_DIR = _TEXTURES_DIR / ".cache"


def _ensure_cache() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _svg_to_png(svg_name: str, png_name: str, size: tuple[int, int]) -> str:
    png_path = _CACHE_DIR / png_name
    if not png_path.exists():
        _ensure_cache()
        svg_path = _TEXTURES_DIR / svg_name
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=size[0],
            output_height=size[1],
        )
    return str(png_path)


def ensure_textures() -> dict[str, str]:
    return {
        "field": _svg_to_png("field.svg", "field.png", (1800, 1200)),
        "goal": _svg_to_png("goal.svg", "goal.png", (512, 512)),
        "ball": _svg_to_png("ball.svg", "ball.png", (512, 256)),
        "sky": _svg_to_png("sky.svg", "sky.png", (2048, 1024)),
    }
