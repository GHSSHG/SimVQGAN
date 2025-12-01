from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class WandbLogger:
    run: Any

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass


def init_wandb(
    project: str,
    run_name: Optional[str],
    config: Dict[str, Any],
    api_key: Optional[str] = None,
    entity: Optional[str] = None,
) -> Optional[WandbLogger]:
    try:
        import wandb
    except Exception as exc:
        print(f"[warn] wandb import failed: {exc}")
        return None
    init_kwargs: Dict[str, Any] = {
        "project": project,
        "name": run_name,
        "config": config,
        "reinit": True,
    }
    if entity:
        init_kwargs["entity"] = entity
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        try:
            wandb.login(key=api_key, relogin=True)
        except TypeError:
            wandb.login(relogin=True)
    run = wandb.init(**init_kwargs)
    if run is None:
        return None
    return WandbLogger(run=run)
