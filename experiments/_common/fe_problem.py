# -*- coding: utf-8 -*-
"""制造解线弹性问题的统一构建: from_box 结构化网格 + 向量 Lagrange 空间 + 各向同性材料.

网格取 ``MESH_SPECS`` 的四种之一, 默认六面体 (``DEFAULT_MESH``); 二维网格配平面应变制造解,
三维网格配无散度多项式制造解. 取 ``mesh_type="tet", p=1`` 时与
``experiments/fa_assembly_capability/run.py`` 的 ``_build_problem_space`` 同源, 保证 fa / ea
两个目录在同一 n 下得到逐位相同的网格、自由度编号与单刚 ``K_e``.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Tuple


class MeshSpec(NamedTuple):
    """一种网格的构建约定.

    Parameters
    ----------
    mesh_class : FEALPy 网格类名, 由 ``from_box`` 生成.
    problem_name : 配套的制造解问题类名 (``soptx.problems.elasticity``).
    hypothesis : 材料本构的降维方式.
    GD : 几何维数.
    extra_order : 积分参数 q 相对 p 的增量, q = p + extra_order; 单纯形为 3,
        张量积单元 q 是每方向点数, 取 1 即精确积出刚度.
    """

    mesh_class: str
    problem_name: str
    hypothesis: str
    GD: int
    extra_order: int


MESH_SPECS: Dict[str, MeshSpec] = {
    "tri": MeshSpec("TriangleMesh", "ExponentialSineManufacturedElasticity2D", "plane_strain", 2, 3),
    "quad": MeshSpec("QuadrangleMesh", "ExponentialSineManufacturedElasticity2D", "plane_strain", 2, 1),
    "tet": MeshSpec("TetrahedronMesh", "DivergenceFreePolynomialElasticity3D", "3D", 3, 3),
    "hex": MeshSpec("HexahedronMesh", "DivergenceFreePolynomialElasticity3D", "3D", 3, 1),
}
DEFAULT_MESH = "hex"

METHOD_NAMES = ("standard", "voigt", "fast")
PROBLEM_NAME = MESH_SPECS[DEFAULT_MESH].problem_name
MESH_TYPE = MESH_SPECS[DEFAULT_MESH].mesh_class
T_TET4 = 288  # tet4 线弹性: 每自由度对应的三元组数 (144 * NC / Ndof 的渐近值)


def integration_order(mesh_type: str, p: int) -> int:
    """网格 ``mesh_type`` 上 p 次空间的积分参数 q, 见 ``MeshSpec.extra_order``."""
    return p + MESH_SPECS[mesh_type].extra_order


def is_cuda(device_str: str) -> bool:
    return device_str.startswith("cuda") or device_str == "gpu"


def setup_cuda(device_str: str) -> Tuple[Any, str]:
    """切换 FEALPy 后端到 PyTorch 并绑定 CUDA 设备; 返回 (torch.device, 规范化设备名)."""
    import torch
    from fealpy.backend import backend_manager as bm

    device_str = "cuda:0" if device_str in ("gpu", "cuda") else device_str
    bm.set_backend("pytorch")
    bm.set_default_device(device_str)
    device = torch.device(device_str)
    torch.cuda.set_device(device.index if device.index is not None else 0)
    return device, device_str


def build_problem_space(
    n: int,
    device: Any = None,
    *,
    mesh_type: str = DEFAULT_MESH,
    p: int = 1,
) -> Tuple[Any, Any, Any, Any]:
    """构建制造解问题、from_box 网格、向量 p 次空间与材料 (需先设置后端).

    Parameters
    ----------
    n : int
        网格每方向段数.
    device : optional
        材料常量所在设备; ``None`` 时取网格所在设备.
    mesh_type : str, optional
        网格类型, 取 ``MESH_SPECS`` 的键之一, 默认六面体.
    p : int, optional
        Lagrange 空间次数, 默认 1.

    Returns
    -------
    problem, mesh, vs, material
    """
    import fealpy.mesh
    import soptx.problems.elasticity
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from soptx.materials import IsotropicLinearElasticMaterial

    if mesh_type not in MESH_SPECS:
        raise ValueError(f"未知的网格类型 {mesh_type!r}, 可选 {tuple(MESH_SPECS)}")
    spec = MESH_SPECS[mesh_type]
    problem = getattr(soptx.problems.elasticity, spec.problem_name)()
    mesh_class = getattr(fealpy.mesh, spec.mesh_class)
    mesh = mesh_class.from_box(list(problem.domain), *(n, ) * spec.GD)
    scalar = LagrangeFESpace(mesh, p=p, ctype="C")
    vs = TensorFunctionSpace(scalar, shape=(-1, spec.GD))
    if device is None:
        from fealpy.backend import backend_manager as bm

        device = bm.get_device(mesh)
    material = IsotropicLinearElasticMaterial(
        hypothesis=spec.hypothesis,
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=device,
    )
    return problem, mesh, vs, material


def import_fe_stack_cpu() -> None:
    """在测量开始前把 FEALPy / SOPTX 相关模块全部导入, 避免 import 开销混入阶段测量."""
    from fealpy.backend import backend_manager as bm

    bm.set_backend("numpy")
    import fealpy.functionspace  # noqa: F401
    import fealpy.mesh  # noqa: F401
    import fealpy.sparse  # noqa: F401
    import soptx.fem.integrators  # noqa: F401
    import soptx.fem.matrix.csr_pattern  # noqa: F401
    import soptx.materials  # noqa: F401
    import soptx.problems.elasticity  # noqa: F401


def mesh_facts(mesh: Any, vs: Any) -> Dict[str, int]:
    return {
        "NC": int(mesh.number_of_cells()),
        "NN": int(mesh.number_of_nodes()),
        "Ndof": int(vs.number_of_global_dofs()),
    }
