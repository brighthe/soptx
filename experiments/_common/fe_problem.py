# -*- coding: utf-8 -*-
"""制造解线弹性问题的统一构建: tet4 网格 + 向量 P1 空间 + 各向同性材料.

与 ``experiments/fa_assembly_capability/run.py`` 的 ``_build_problem_space`` 同源, 保证
fa / ea 两个目录在同一 n 下得到逐位相同的网格、自由度编号与单刚 ``K_e``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

METHOD_NAMES = ("standard", "voigt", "fast")
PROBLEM_NAME = "DivergenceFreePolynomialElasticity3D"
MESH_TYPE = "TetrahedronMesh"
T_TET4 = 288  # tet4 线弹性: 每自由度对应的三元组数 (144 * NC / Ndof 的渐近值)


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


def build_problem_space(n: int, device: Any = None) -> Tuple[Any, Any, Any, Any]:
    """构建制造解问题、tet4 网格、向量 P1 空间与材料 (需先设置后端).

    Parameters
    ----------
    n : int
        网格每方向段数.
    device : optional
        材料常量所在设备; ``None`` 时取网格所在设备.

    Returns
    -------
    problem, mesh, vs, material
    """
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import TetrahedronMesh
    from soptx.materials import IsotropicLinearElasticMaterial
    from soptx.problems.elasticity import DivergenceFreePolynomialElasticity3D

    problem = DivergenceFreePolynomialElasticity3D()
    mesh = TetrahedronMesh.from_box(list(problem.domain), nx=n, ny=n, nz=n)
    scalar = LagrangeFESpace(mesh, p=1, ctype="C")
    vs = TensorFunctionSpace(scalar, shape=(-1, 3))
    if device is None:
        from fealpy.backend import backend_manager as bm

        device = bm.get_device(mesh)
    material = IsotropicLinearElasticMaterial(
        hypothesis="3D",
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
