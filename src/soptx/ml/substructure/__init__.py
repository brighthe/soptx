"""子结构机器学习的数据、训练与 artifact 契约。"""

from .artifacts import (
    ArchitectureSignature,
    ArtifactCompatibilityError,
    ModelSignature,
    load_checkpoint,
    load_legacy_state_dict,
    save_checkpoint,
)
from .nets import PIMLSurrogateNet, ShapeFunctionSurrogateNet
from .sampling import (
    SAMPLER_VERSION,
    DensitySamples,
    DensitySamplingConfig,
    SamplingFractions,
    sample_density_fields,
)
from .training import TrainingConfig, TrainingResult, train_surrogate

__all__ = [
    "SAMPLER_VERSION",
    "ArchitectureSignature",
    "ArtifactCompatibilityError",
    "DensitySamples",
    "DensitySamplingConfig",
    "ModelSignature",
    "PIMLSurrogateNet",
    "SamplingFractions",
    "ShapeFunctionSurrogateNet",
    "TrainingConfig",
    "TrainingResult",
    "load_checkpoint",
    "load_legacy_state_dict",
    "sample_density_fields",
    "save_checkpoint",
    "train_surrogate",
]
