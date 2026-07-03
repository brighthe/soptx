"""PIML 多尺度预测原型 (路线①·子结构静力缩聚).

模块结构 (对应 soptx-piml-multiscale-integration-plan §2.2):

- coarse_fine_mesh:       T1 粗/细两级网格与自由度映射
- multiscale_shape:       T2 参考子结构模板与静力缩聚 (Schur 补)
- piml_predictor:         T4 预测器抽象 (Exact / Mock)
- equivalent_stiffness:   T3 接口缩聚组装 + T5 单步前向闭环
- fullscale_reference:    V1 全尺度参照 (全局 Schur 补 / 直解)

原型阶段仅支持 numpy 后端; 与 matrix-free 的协同 (K̂_s^j 喂给全局
matrix-free 作用) 属后续阶段, 见 ai/common/progress-frame7_piml.md「阶段二/三 接入设计」.
"""

from .coarse_fine_mesh import CoarseFineMeshPair
from .multiscale_shape import SubstructureTemplate, SubstructureOperator
from .piml_predictor import MultiscalePredictor, ExactPredictor, MockPredictor
from .equivalent_stiffness import InterfaceCondensedSystem
from .fullscale_reference import (
    assemble_fullscale_stiffness,
    fullscale_schur_complement,
    solve_with_dirichlet,
)

__all__ = [
    "CoarseFineMeshPair",
    "SubstructureTemplate",
    "SubstructureOperator",
    "MultiscalePredictor",
    "ExactPredictor",
    "MockPredictor",
    "InterfaceCondensedSystem",
    "assemble_fullscale_stiffness",
    "fullscale_schur_complement",
    "solve_with_dirichlet",
]

# T4b TrainedPredictor 依赖 torch; 原型其余部分 numpy-only, 故 torch 缺失时
# 不影响 Exact/Mock 前向管道与 V1/V2 测试导入.
try:  # pragma: no cover - 依赖可选后端
    from .trained_predictor import (
        KsMLP,
        Standardizer,
        TrainedPredictor,
        train_ks_mlp,
        make_training_data,
        save_predictor,
        load_predictor,
    )

    __all__ += [
        "KsMLP",
        "Standardizer",
        "TrainedPredictor",
        "train_ks_mlp",
        "make_training_data",
        "save_predictor",
        "load_predictor",
    ]
except ImportError:  # torch 未安装
    pass
