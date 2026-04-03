from .encoder import SimVQEncoder1D, SimVQResBlock1D
from .decoder import SimVQDecoder1D
from .factory import build_audio_model
from .model import SimVQAudioModel
from .quantize import SimVQ1D
from .transformer import LocalTransformerBlock1D, SwinTransformerBlock1D, TransformerBlock1D

# Backwards-compatible aliases for downstream scripts
Encoder = SimVQEncoder1D
Decoder1D = SimVQDecoder1D
AudioCodecModel = SimVQAudioModel

__all__ = [
    "SimVQEncoder1D",
    "SimVQResBlock1D",
    "SimVQDecoder1D",
    "SimVQAudioModel",
    "build_audio_model",
    "SimVQ1D",
    "LocalTransformerBlock1D",
    "SwinTransformerBlock1D",
    "TransformerBlock1D",
]
