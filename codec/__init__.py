"""codec: Modularized JAX/Flax 1D audio codec training package.

The modules split data loading, model definitions, losses, training steps,
and utilities so the project can run cleanly on local workstations instead
of notebook-only workflows.

See scripts/train.py for a CLI entrypoint.
"""

__all__ = [
    "data",
    "jaxlayers",
    "audio",
    "models",
    "train",
    "utils",
]
