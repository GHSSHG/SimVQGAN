from .encoder import SimVQEncoder1D, SimVQResBlock1D
from .decoder import SimVQDecoder1D
from .model import SimVQAudioModel
from .patchgan import PatchDiscriminator1D
from .quantize import SimVQ1D
from .transformer import TransformerBlock1D

# Backwards-compatible aliases for downstream scripts
Encoder = SimVQEncoder1D
Decoder1D = SimVQDecoder1D
AudioCodecModel = SimVQAudioModel

__all__ = [
    "SimVQEncoder1D",
    "SimVQResBlock1D",
    "SimVQDecoder1D",
    "SimVQAudioModel",
    "PatchDiscriminator1D",
    "SimVQ1D",
    "TransformerBlock1D",
]
