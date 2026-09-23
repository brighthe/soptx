# -*- coding: utf-8 -*-
"""五种矩阵组装层级 (stored-B / EA / FA / PA / shared-Ke) 的共用问题、算子与计量.

本模块抽自 ``experiments/assembly_level_capability/run.py``, 供正确性目录
``assembly_level_consistency/`` 与性能目录 ``assembly_level_capability/`` 共用, 保证两边
量到的是同一套网格、材料、积分与算子实现。

五种方案共用同一网格 (单位立方体 n^3 六面体 Q1)、同一材料 (E = 1, nu = 0.3)、同一积分
(q = 2, 8 点)、同一交错自由度布局 (dof = 3 * node + comp)。其中 ea / fa / pa 由
``soptx.fem.levels.create_level`` 构造, 量的就是生产栈本身; stored-b (臧昕禹推断方案) 与
shared-ke (均匀网格共享一份 K_e) 没有生产对应物, 是本模块自带的对照实现。
"""

from __future__ import annotations

import os
import platform
import resource
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy

DEFAULT_MEMORY_TOTAL = 47.04 * 2**30  # 本机 WSL 可用内存上限 (47.04 GiB)
PROBLEM = "LinearElasticity3D_UnitCube"
MESH_TYPE = "HexahedronMesh"
QUADRATURE_ORDER = 2  # hex Q1 精确积分刚度所需的 2x2x2 Gauss 点; 积分器默认 q = p + 3 会给 64 点

# 每单元理论存储 (double 个数, 不含 cell2dof 的 24 个 int64)
THEORY_NUMBERS_PER_CELL: Dict[str, int] = {
    "stored-b": 8 * 6 * 24 + 8 * 6 * 6 + 8,  # B_q (6x24) + D_q (6x6) + w_q detJ_q, 8 个积分点
    "ea": 24 * 24,
    "fa": 0,  # 按实测 nnz 折算, 见 LevelOperator.numbers_per_cell
    "pa": 8 * 9 + 8,  # J^{-1}_q (3x3) + w_q detJ_q
    "shared-ke": 0,  # 全网格共享一份 24x24
}


# -----------------------------------------------------------------------------
# 1. 计量
# -----------------------------------------------------------------------------

def get_peak_rss_bytes() -> int:
    """获取当前进程生命周期的最高内存水位 (ru_maxrss)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss
    return usage.ru_maxrss * 1024


def get_current_rss_bytes() -> int:
    """读取当前进程此刻的常驻内存 (Linux /proc/self/statm 第 2 列, 单位页).

    与 ru_maxrss 不同, 这是时点值而非高水位, 两个时点相减即得中间步骤的净增量;
    非 Linux 平台回退为 ru_maxrss。
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return get_peak_rss_bytes()


def collect_env(thread_env: Dict[str, str]) -> Dict[str, Any]:
    """记录运行环境: 版本、线程环境变量与 BLAS 信息.

    Parameters
    ----------
    thread_env : 调用方 ``config.THREAD_ENV``, 只取其键名去读当前进程的实际取值。
    """
    blas = {}
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        blas = {k: blas.get(k) for k in ("name", "version") if k in blas}
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "threads": {k: os.environ.get(k) for k in thread_env},
        "blas": blas,
    }


# -----------------------------------------------------------------------------
# 2. 物理构件
# -----------------------------------------------------------------------------

def setup_problem(n: int):
    """构建单位立方体 n^3 六面体 Q1 网格、位移张量空间、材料与积分数据.

    Returns
    -------
    ctx : dict
        含 mesh, scalar_space, tensor_space, material, integrator, D0 (6, 6),
        bcs, ws (NQ,), cell2dof (NC, 24) int64, n_cells, n_dofs。
    """
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import HexahedronMesh
    from soptx.fem import LinearElasticIntegrator
    from soptx.materials import IsotropicLinearElasticMaterial

    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    scalar_space = LagrangeFESpace(mesh, p=1, ctype="C")
    tensor_space = TensorFunctionSpace(scalar_space=scalar_space, shape=(-1, 3))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1.0, poisson_ratio=0.3)
    D0 = np.ascontiguousarray(np.asarray(material.elastic_matrix())[0, 0], dtype=np.float64)

    qf = mesh.quadrature_formula(QUADRATURE_ORDER)
    bcs, ws = qf.get_quadrature_points_and_weights()
    ws = np.asarray(ws, dtype=np.float64)

    cell2dof = np.ascontiguousarray(np.asarray(tensor_space.cell_to_dof()), dtype=np.int64)

    # ea / fa / pa 的生产层级统一从 (space, integrator) 构造。q 必须显式给:
    # PartialAssembly.build 读的是 integrator.q, 为 None 时回落到 p + 3, hex Q1 会变成
    # 64 个积分点而不是本实验约定的 8 个。
    integrator = LinearElasticIntegrator(material=material, q=QUADRATURE_ORDER, method="fast")

    return {
        "mesh": mesh,
        "scalar_space": scalar_space,
        "tensor_space": tensor_space,
        "material": material,
        "integrator": integrator,
        "D0": D0,
        "bcs": bcs,
        "ws": ws,
        "cell2dof": cell2dof,
        "n_cells": int(mesh.number_of_cells()),
        "n_dofs": int(tensor_space.number_of_global_dofs()),
    }


def reference_K_e(ctx) -> np.ndarray:
    """用 soptx 的 fast 路径算参考单元刚度 (NC, 24, 24).

    Notes
    -----
    与 create_level 共用 ctx["integrator"] 是安全的: assembly 每次调用都重算, 积分子
    本身不留缓存, 因此层级的 build 不会白捡这一次结果, matvec 面板 build_seconds 的
    口径不受影响。
    """
    integrator = ctx["integrator"]
    K_e = integrator.assembly(ctx["tensor_space"])
    return np.ascontiguousarray(np.asarray(K_e), dtype=np.float64)


def geometry(ctx) -> Tuple[np.ndarray, np.ndarray]:
    """逐单元逐积分点的 Jacobi 矩阵 J (NC, NQ, 3, 3) 与 w_q |det J_q| (NC, NQ)."""
    mesh = ctx["mesh"]
    J = np.asarray(mesh.entity_view("cell").jacobi_matrix(ctx["bcs"]), dtype=np.float64)
    detJ = np.abs(np.linalg.det(J))
    wdetJ = np.ascontiguousarray(ctx["ws"][None, :] * detJ)
    return J, wdetJ


# -----------------------------------------------------------------------------
# 3. 五种方案的算子
# -----------------------------------------------------------------------------

class ElementOperator:
    """单元级算子的公共骨架: gather -> 局部作用 -> scatter-add.

    子类只实现 ``_apply_local(x_e) -> y_e``; ``persistent_arrays`` 列出常驻数组用于计字节。
    """

    scheme = ""

    def __init__(self, cell2dof: np.ndarray):
        self.cell2dof = cell2dof
        self.flat_cell2dof = cell2dof.reshape(-1)
        self.n_dofs = int(cell2dof.max()) + 1

    def persistent_arrays(self) -> List[np.ndarray]:
        return [self.cell2dof]

    def persistent_bytes(self) -> int:
        return int(sum(a.nbytes for a in self.persistent_arrays()))

    @property
    def numbers_per_cell(self) -> float:
        return float(THEORY_NUMBERS_PER_CELL[self.scheme])

    def _apply_local(self, x_e: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_e = x[self.cell2dof]
        y_e = self._apply_local(x_e)
        y = np.zeros_like(x)
        np.add.at(y, self.flat_cell2dof, y_e.ravel())
        return y


class StoredBOperator(ElementOperator):
    """逐单元逐积分点存物理 B_q (6x24)、D_q (6x6) 与 w_q detJ_q (臧昕禹推断方案)."""

    scheme = "stored-b"

    def __init__(self, ctx):
        super().__init__(ctx["cell2dof"])
        gphi_x = np.asarray(ctx["scalar_space"].grad_basis(ctx["bcs"], variable="x"), dtype=np.float64)
        B = ctx["material"].strain_matrix(dof_priority=ctx["tensor_space"].dof_priority, gphi=gphi_x)
        self.B = np.ascontiguousarray(np.asarray(B), dtype=np.float64)  # (NC, NQ, 6, 24)
        del gphi_x, B
        NC, NQ = self.B.shape[:2]
        # 学生方案把 D_q 逐点落盘; 这里用真实拷贝 (不是 broadcast 视图) 复现其内存占用。
        self.D = np.ascontiguousarray(np.broadcast_to(ctx["D0"], (NC, NQ, 6, 6)))
        _, self.wdetJ = geometry(ctx)

    def persistent_arrays(self):
        return [self.cell2dof, self.B, self.D, self.wdetJ]

    def _apply_local(self, x_e):
        e = np.einsum("cqij,cj->cqi", self.B, x_e)
        s = np.einsum("cqkl,cql->cqk", self.D, e)
        s *= self.wdetJ[..., None]
        return np.einsum("cqki,cqk->ci", self.B, s)


class SharedKeOperator(ElementOperator):
    """均匀网格 + 均匀材料: 全网格共享一份 K_e (24x24), 局部作用是一次 dgemm."""

    scheme = "shared-ke"

    def __init__(self, ctx, K_e: np.ndarray):
        super().__init__(ctx["cell2dof"])
        self.K_e0 = np.ascontiguousarray(K_e[0])
        self.K_e0_T = np.ascontiguousarray(self.K_e0.T)
        # 共享前提的实测证据: 所有单元 K_e 与第 0 个的最大偏差
        self.max_deviation = float(np.max(np.abs(K_e - self.K_e0[None])))

    def persistent_arrays(self):
        return [self.cell2dof, self.K_e0]

    def _apply_local(self, x_e):
        return x_e @ self.K_e0_T


class LevelOperator:
    """把 soptx.fem.levels 的生产层级 (ea / fa / pa) 包成 harness 的算子接口.

    这三个方案不再在本模块里手写一遍: 基准量到的必须是 soptx 真正会跑的那份实现, 否则
    改了生产类基准不会发现, 报出的内存与耗时也不是生产栈的口径。

    Parameters
    ----------
    scheme : 方案 id, 同时也是层级注册键 ("ea" / "fa" / "pa")。
    ctx : ``setup_problem`` 的返回值, 用其中的 tensor_space 与 integrator。

    Notes
    -----
    FA 的 MatVec 走 ``FullAssembly.__matmul__`` -> fealpy ``CSRTensor.matmul`` ->
    ``bm.csr_spmm``, 最终落到 ``scipy.sparse._sparsetools.csr_matvec``, 与
    ``scipy.sparse.csr_matrix @ x`` 是同一个 C 例程; 多出的只有每次调用几微秒的 Python
    转发, n = 48 起单次 MatVec 已是毫秒量级。EA / PA 的 scatter-add 走
    ``ElementRestriction.scatter_add`` -> ``bm.index_add`` -> ``np.add.at``, 与
    stored-b / shared-ke 手写的那条是同一个 numpy 调用。五方案因此仍然可比。
    """

    def __init__(self, scheme: str, ctx):
        from soptx.fem.levels import create_level

        self.scheme = scheme
        self.level = create_level(scheme,
                                  space=ctx["tensor_space"],
                                  integrator=ctx["integrator"])
        self.n_cells = ctx["n_cells"]
        self.n_dofs = ctx["n_dofs"]

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.level @ x

    def persistent_bytes(self) -> int:
        return int(self.level.persistent_bytes())

    @property
    def numbers_per_cell(self) -> float:
        if self.scheme == "fa":
            # values + 列索引按 8 字节一个数折算到每单元
            matrix = self.level.matrix
            return float((matrix.values.nbytes + matrix.col.nbytes) / 8 / self.n_cells)
        return float(THEORY_NUMBERS_PER_CELL[self.scheme])

    @property
    def nnz(self) -> int:
        """FA 专有: 全局矩阵非零元个数"""
        return int(self.level.matrix.values.shape[0])

    @property
    def index_dtype(self) -> str:
        """FA 专有: CSR 列索引的 dtype (int32 与 int64 差一倍索引内存)"""
        return str(self.level.matrix.col.dtype)


def build_operator(scheme: str, ctx, K_e: Optional[np.ndarray] = None):
    """按方案 id 构造算子; 只有 shared-ke 还需要外部传入的参考 K_e."""
    if scheme in ("ea", "fa", "pa"):
        return LevelOperator(scheme, ctx)
    if scheme == "stored-b":
        return StoredBOperator(ctx)
    if K_e is None:
        raise ValueError(f"方案 '{scheme}' 需要参考 K_e")
    if scheme == "shared-ke":
        return SharedKeOperator(ctx, K_e)
    raise ValueError(f"未知方案: {scheme}")


# -----------------------------------------------------------------------------
# 4. 产物头尾与格式化
# -----------------------------------------------------------------------------

def common_header(case_id: str, panel: str, role: str, scheme: str, n: int, ctx,
                  thread_env: Dict[str, str]) -> Dict[str, Any]:
    """JSON 产物的公共头部: 工况标识、网格规模与运行环境."""
    return {
        "case_id": case_id,
        "panel": panel,
        "role": role,
        "problem": PROBLEM,
        "mesh_type": MESH_TYPE,
        "grid": f"{n}^3",
        "n": n,
        "n_cells": ctx["n_cells"] if ctx else 0,
        "n_dofs": ctx["n_dofs"] if ctx else 0,
        "scheme": scheme,
        "device": "CPU",
        "device_raw": "cpu",
        "quadrature_order": QUADRATURE_ORDER,
        "env": collect_env(thread_env),
    }


def memory_tail(peak_bytes: int, n_dofs: int) -> Dict[str, Any]:
    """JSON 产物的内存尾部: 峰值 RSS 与按自由度折算的单位成本、容量上限."""
    unit_cost_bytes = peak_bytes / n_dofs if n_dofs > 0 else 0.0
    return {
        "peak_memory_bytes": peak_bytes,
        "peak_memory_mib": peak_bytes / (1024**2),
        "unit_cost_bytes_per_dof": unit_cost_bytes,
        "unit_cost_kb_per_dof": unit_cost_bytes / 1000,
        "capacity_ceiling_47g_dofs": int(DEFAULT_MEMORY_TOTAL / unit_cost_bytes) if unit_cost_bytes > 0 else 0,
    }


def fmt_bytes(b: float) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GiB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MiB"
    return f"{b / 1024:.1f} KiB"


def fmt_seconds(s: float) -> str:
    return f"{s:.3f} s" if s >= 1.0 else f"{s * 1000:.2f} ms"
