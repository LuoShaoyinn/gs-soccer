from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from .advanced_dribble import AdvancedDribblePolicy
from .go_to_ball_pid import GoToBallPIDPolicy
from .zero import ZeroPolicy


def available_policy_modules() -> list[str]:
    module_names: list[str] = []
    for module in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if module.name.startswith("_"):
            continue
        module_names.append(module.name)
    return sorted(module_names)


def build_policy(module_name: str, **kwargs: Any) -> Any:
    module = importlib.import_module(f"{__name__}.{module_name}")
    policy_cls = getattr(module, "Policy", None)
    if policy_cls is None:
        raise RuntimeError(f"Policy module '{module_name}' does not export `Policy`")
    return policy_cls(**kwargs)


__all__ = [
    "AdvancedDribblePolicy",
    "GoToBallPIDPolicy",
    "ZeroPolicy",
    "available_policy_modules",
    "build_policy",
]
