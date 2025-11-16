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


def init_wandb(project: str, run_name: Optional[str], config: Dict[str, Any], api_key: Optional[str] = None) -> Optional[WandbLogger]:
    try:
        import wandb
    except Exception as exc:
        print(f"[warn] wandb import failed: {exc}")
        return None
    if api_key:
        os.environ.setdefault("WANDB_API_KEY", api_key)
    run = wandb.init(project=project, name=run_name, config=config, reinit=True)
    if run is None:
        return None
    return WandbLogger(run=run)
