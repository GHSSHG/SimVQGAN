"""codec: Modularized JAX/Flax 1D audio codec training package.

This package is an extraction from a large Colab script (colab.py).
It splits data loading, model definitions, losses, training steps,
and utilities into focused modules so you can reuse and maintain code
outside of notebooks.

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

