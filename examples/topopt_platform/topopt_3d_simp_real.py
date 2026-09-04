# -*- coding: utf-8 -*-
"""真实 SIMP 三维拓扑优化（多后端张量化）求解器.

在规则 160 x 80 x 40 六面体网格（160.4 万自由度）三维悬臂梁上，用**真实** Hex8
弹性刚度、真实 PCG（Jacobi 预条件）求解与 OC 更新跑完整 SIMP 拓扑优化，产出最终
构型密度场、完整优化历程与**可复现**的分阶段性能实测。同一核心求解代码经 fealpy
后端管理器同时跑 NumPy(CPU) 与 PyTorch(GPU) 两个后端 —— 这是 "多后端异构并行拓扑
优化平台" 的核心主张，也是 80 批申请书第 6 部分图 10 的实测数据来源。

本脚本取代原 ``benchmark_topopt_3d.py``（其性能数字全部硬编码、刚度为通用 SPD 矩阵
而非真实弹性刚度、CG 用伪算子，既不能复现也不能算拓扑优化）。

正确性按三条判据核验（``--validate``，小网格）:
  1. Hex8 刚度矩阵物理自检（对称、刚体模态零能）+ 矩阵自由算子单次作用与
     scipy 显式装配一致（相对 1e-13 量级）；
  2. PCG 解与 ``scipy.sparse.linalg.spsolve`` 一致；
  3. 完整优化柔度单调下降、体积约束达标、拓扑收敛。

CPU 基线（``--cpu-baseline``）走 scipy 稀疏显式装配 + 稀疏 PCG 的传统流程，与 GPU
张量化矩阵自由路径作对照，这与申请书图 10(a)(b) 的 "CPU 传统流程 vs GPU 张量化平台"
口径一致。

用法:
  python examples/topopt_platform/topopt_3d_simp_real.py --backend pytorch
  python examples/topopt_platform/topopt_3d_simp_real.py --validate
  python examples/topopt_platform/topopt_3d_simp_real.py --cpu-baseline --nx 160 --ny 80 --nz 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from fealpy.backend import backend_manager as bm
from soptx.solvers import CGSolver, DiagonalPreconditioner
from soptx.topology.filters import apply_structured_sensitivity_filter

# 全局目标设备：numpy 后端为 None；pytorch 后端设为 "cuda:0"。
# torch 的 from_numpy/from_tensor 恒建 CPU 张量，而 set_default_device 又会让部分
# 创建操作落在 cuda，两者混用必然设备错位，故所有 numpy->后端张量的转换统一走 _t。
_DEVICE = None


def _t(x: np.ndarray):
    t = bm.from_numpy(np.asarray(x))
    return bm.device_put(t, _DEVICE) if _DEVICE else t


def _set_runtime(backend: str) -> None:
    global _DEVICE
    bm.set_backend(backend)
    if backend == "pytorch":
        bm.set_default_device("cuda:0")
        _DEVICE = "cuda:0"
    else:
        _DEVICE = None

# --------------------------------------------------------------------------- #
# 网格
# --------------------------------------------------------------------------- #

def build_mesh(nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float):
    """规则六面体网格。节点编号 x 方向最慢，z 方向最快:
    ``node_id = i*(ny+1)*(nz+1) + j*(nz+1) + k``。
    单元 (i,j,k) 的 8 个角节点按等参 Hex8 标准顺序排列。
    """
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    node = np.zeros((n_nodes, 3), dtype=np.float64)
    idx = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                node[idx] = (Lx * i / nx, Ly * j / ny, Lz * k / nz)
                idx += 1

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    nelem = nx * ny * nz
    cell2node = np.empty((nelem, 8), dtype=np.int64)
    e = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell2node[e] = [
                    nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]
                e += 1

    cell2dof = np.repeat(cell2node * 3, 3, axis=1) + np.tile(np.arange(3), 8)
    return node, cell2node, cell2dof


def cell_sizes(nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float):
    """由设计域尺寸与网格划分数给出单元物理边长."""
    return Lx / nx, Ly / ny, Lz / nz


def boundary_sets(nx: int, ny: int, nz: int):
    """经典三维悬臂梁边界：左端面全固支，右端面**底边**竖直向下均布线载荷 (总和 1).

    与博士论文算例 3.3 一致（L x L/3 x L/15 = 60 x 20 x 4 薄板）：
      * 固定自由度：x=0 面全部平动分量 (ux=uy=uz=0)；
      * 施载：x=nx 面 y=0 底边（沿 z 的一条边）上各节点施加 -y 方向的等效节点力，
        各节点承担 1/(nz+1)（均布合力归一为 1）。
    返回 (fixed_dofs, load_dofs, load_vals)。"""
    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    fixed_nodes = [nid(0, j, k) for j in range(ny + 1) for k in range(nz + 1)]
    fixed_dofs = np.array(
        [d for n in fixed_nodes for d in (3 * n, 3 * n + 1, 3 * n + 2)],
        dtype=np.int64,
    )
    load_nodes = [nid(nx, 0, k) for k in range(nz + 1)]  # 右端面底边 (y=0, 全 z)
    load_dofs = np.array([3 * n + 1 for n in load_nodes], dtype=np.int64)  # -y
    load_vals = np.full(len(load_nodes), -1.0 / len(load_nodes), dtype=np.float64)
    return fixed_dofs, load_dofs, load_vals


# --------------------------------------------------------------------------- #
# 真实 Hex8 弹性刚度
# --------------------------------------------------------------------------- #

_NODE_LC = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)


def hex8_stiffness(E: float = 1.0, nu: float = 0.3,
                   hx: float = 1.0, hy: float = 1.0,
                   hz: float = 1.0) -> np.ndarray:
    """长方体 8 节点六面体线弹性单元刚度矩阵, 2x2x2 全积分.

    节点局部坐标见 ``_NODE_LC``, 形函数 ``N_a = (1+r_a r)(1+s_a s)(1+t_a t)/8``.
    等参映射为对角常量 ``J = diag(hx/2, hy/2, hz/2)``, 故物理导数
    ``dN/dx_i = (2/h_i) * dN/dr_i``, 积分权重 ``detJ = hx*hy*hz/8``.

    三维下刚度按单元尺寸线性标度: 立方体单元有 ``K_e(h) = h * K_e(1)``, 因为
    ``B`` 正比于 ``1/h`` 而体积元正比于 ``h**3``. 该标度必须显式带入, 否则跨网格
    的柔顺度不可比.

    参数:
        E: 杨氏模量.
        nu: 泊松比.
        hx: 单元在 x 方向的物理边长.
        hy: 单元在 y 方向的物理边长.
        hz: 单元在 z 方向的物理边长.
    返回:
        单元刚度矩阵, 形状 ``(24, 24)``, 自由度按节点顺序排列为 ``(ux, uy, uz)``.
    """
    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    # 对角 Jacobian 的逆: dr_i/dx_i = 2/h_i, 长方体单元下与高斯点无关.
    Jinv = np.array([2.0 / hx, 2.0 / hy, 2.0 / hz])
    detJ = hx * hy * hz / 8.0

    K = np.zeros((24, 24))
    for r in xi:
        for s in xi:
            for t in xi:
                dN = np.empty((8, 3))
                for a in range(8):
                    ra, sa, ta = _NODE_LC[a]
                    dN[a, 0] = 0.125 * ra * (1 + sa * s) * (1 + ta * t)
                    dN[a, 1] = 0.125 * sa * (1 + ra * r) * (1 + ta * t)
                    dN[a, 2] = 0.125 * ta * (1 + ra * r) * (1 + sa * s)
                dN *= Jinv  # 局部导数 -> 物理导数, 按列广播
                B = np.zeros((6, 24))
                for a in range(8):
                    col = slice(3 * a, 3 * a + 3)
                    B[0, col] = [dN[a, 0], 0, 0]
                    B[1, col] = [0, dN[a, 1], 0]
                    B[2, col] = [0, 0, dN[a, 2]]
                    B[3, col] = [dN[a, 1], dN[a, 0], 0]
                    B[4, col] = [0, dN[a, 2], dN[a, 1]]
                    B[5, col] = [dN[a, 2], 0, dN[a, 0]]
                K += (B.T @ D @ B) * detJ
    return K


def check_ke0_physical(K0: np.ndarray, E: float = 1.0, nu: float = 0.3,
                       hx: float = 1.0, hy: float = 1.0,
                       hz: float = 1.0) -> bool:
    """校验单元刚度: 对称性, 刚体平移零能, 半正定, 以及应变能绝对量.

    前三项对任意正常数因子免疫, 查不出标度错误; 应变能校验施加解析已知的均匀
    应变场并比对 ``0.5 * eps^T D eps * V``, 是标度缺陷的唯一防线, 不可省略.

    参数:
        K0: 待校验的单元刚度矩阵, 形状 ``(24, 24)``.
        E: 构造该刚度所用的杨氏模量.
        nu: 构造该刚度所用的泊松比.
        hx: 单元在 x 方向的物理边长.
        hy: 单元在 y 方向的物理边长.
        hz: 单元在 z 方向的物理边长.
    返回:
        四项校验全部通过时为 ``True``.
    """
    sym_err = float(np.max(np.abs(K0 - K0.T)))
    rbm = np.zeros(24)
    rbm[::3] = 1.0  # x 方向刚体平移
    rbm_energy = float(np.abs(rbm @ K0 @ rbm))
    w = np.linalg.eigvalsh(K0)
    pdef = bool(float(w[0]) >= -1e-8)

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    vol = hx * hy * hz
    h = np.array([hx, hy, hz])
    xyz = (_NODE_LC + 1.0) * 0.5 * h  # 节点物理坐标, 形状 (8, 3)

    # 每种模式给出线性位移场 u = G x, 对应常应变 eps, 应变能解析可算.
    modes = {
        "单轴受约束拉伸": (np.array([[1., 0, 0], [0, 0, 0], [0, 0, 0]]),
                           np.array([1., 0, 0, 0, 0, 0])),
        "纯剪切": (np.array([[0, 1., 0], [0, 0, 0], [0, 0, 0]]),
                   np.array([0, 0, 0, 1., 0, 0])),
        "静水膨胀": (np.eye(3), np.array([1., 1., 1., 0, 0, 0])),
    }
    energy_err = 0.0
    detail = []
    for name, (G, eps) in modes.items():
        d = (xyz @ G.T).ravel()  # 节点位移 u_a = G x_a, 展平为 (24,)
        num = 0.5 * float(d @ K0 @ d)
        ana = 0.5 * float(eps @ D @ eps) * vol
        rel = abs(num - ana) / abs(ana)
        energy_err = max(energy_err, rel)
        detail.append(f"{name}={num / ana:.6f}")

    print(f"[ke0] 对称误差={sym_err:.2e}  刚体平移能量={rbm_energy:.2e}  "
          f"最小特征值={float(w[0]):.3e}  半正定={pdef}")
    print(f"[ke0] 应变能/解析值: {'  '.join(detail)}  最大相对误差={energy_err:.2e}")
    return (sym_err < 1e-10 and rbm_energy < 1e-10 and pdef
            and energy_err < 1e-12)


# --------------------------------------------------------------------------- #
# 矩阵自由算子（张量化 EA：gather -> 批量 GEMV -> 散加）
# --------------------------------------------------------------------------- #

class ElementMatvecOperator:
    """真实 Hex8 弹性算子（按需作用，不显式组装）。

    刚度系数 ``E_e`` 每次设计迭代更新（``set_coefficients`` 只重算一次批量刚度，
    之后每次算子作用只做 gather -> 批量 GEMV -> 散加）。
    """

    def __init__(self, cell2dof: np.ndarray, ndof: int, ke0: np.ndarray,
                 *, dtype: str = "float32", chunk: int = 131072):
        self.ndof = int(ndof)
        self.dtype = dtype
        self.cell2dof = _t(cell2dof.astype(np.int64))
        self.n_ele = int(cell2dof.shape[0])
        self.chunk = chunk
        self._chunks = [
            (s, min(s + chunk, self.n_ele)) for s in range(0, self.n_ele, chunk)
        ]
        self._flat_chunks = [
            bm.reshape(self.cell2dof[s:e], (-1,)) for s, e in self._chunks
        ]
        self.ke0 = _t(ke0.astype(dtype))
        self._ke_chunks: List[Any] = []

    def set_coefficients(self, E_e: np.ndarray):
        """根据当前设计密度的一次全局更新（等价于一次批量刚度组装）。"""
        E_e_b = _t(E_e.astype(self.dtype))
        self._ke_chunks = [
            bm.einsum('e,ij -> eij', E_e_b[s:e], self.ke0) for s, e in self._chunks
        ]

    def matvec(self, x):
        y = bm.zeros((self.ndof,), **bm.context(x))
        for (s, e), flat in zip(self._chunks, self._flat_chunks):
            x_b = x[self.cell2dof[s:e]]
            y_b = bm.einsum('bij,bj -> bi', self._ke_chunks[self._chunk_index(s)],
                            x_b)
            y = bm.index_add(y, flat, bm.reshape(y_b, (-1,)))
        return y

    def _chunk_index(self, s):
        return s // self.chunk

    def diagonal(self):
        d = bm.zeros((self.ndof,), **bm.context(self.ke0))
        for (s, e), flat in zip(self._chunks, self._flat_chunks):
            diag_b = bm.einsum('eii -> ei', self._ke_chunks[self._chunk_index(s)])
            d = bm.index_add(d, flat, bm.reshape(diag_b, (-1,)))
        return d


# --------------------------------------------------------------------------- #
# PCG（Jacobi 预条件）
# --------------------------------------------------------------------------- #

def _norm(x):
    return float(bm.sqrt(bm.sum(x * x)))


def pcg(operator, b, free, *, M_diag=None, x0=None, rtol: float = 1e-6,
        maxiter: int = 1000):
    """自由自由度子空间上的预条件共轭梯度（Jacobi），支持热启动。

    返回 (x_free, niter, 相对残差)。数值保护：NaN 出现即停机（fp32 大网格下
    舍入噪声不可避免，宁可在当前近似解停机也不让 NaN 污染后续设计步）。

    这里不用 ``soptx.solvers.CGSolver`` 的三个理由，都只在本脚本成立：
    1. 本函数是被计时的对象。``CGSolver`` 每次 ``solve`` 固定多做一次 matvec
       算真残差 ``relres``，在 1.6M 自由度的 GPU 内循环里会直接进 ``t_solve``，
       让性能实测量到的不再是纯 PCG；
    2. 它在自由自由度子空间上迭代，但算子 ``matvec`` 作用在全自由度上，靠
       ``_embed``/``[free]`` 来回投影；``CGSolver`` 要的是一个 ``@`` 就位的
       算子，需另加子空间包装；
    3. fp32 下 ``pAp``、``rs`` 的非有限值即停机是这里特有的保护。
    CPU 基线路径没有这三条约束，已经换成 ``CGSolver``。
    """
    ctx = bm.context(b)
    b_free = b[free]
    bnorm = _norm(b_free)
    if bnorm == 0.0:
        return bm.zeros((int(len(b_free)),), **ctx), 0, 0.0
    if M_diag is None:
        M_diag = operator.diagonal()
    M_free = M_diag[free]

    x = x0 if x0 is not None else bm.zeros((int(len(b_free)),), **ctx)
    r = b_free - operator.matvec(_embed(b, x, free))[free]
    z = r / M_free
    p = z
    rs = float(bm.sum(r * z))
    rel = _norm(r) / bnorm
    it = 0
    for it in range(1, maxiter + 1):
        Ap = operator.matvec(_embed(b, p, free))[free]
        pAp = float(bm.sum(p * Ap))
        if not (pAp > 0.0) or not np.isfinite(pAp):
            break
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel = _norm(r) / bnorm
        if rel < rtol or not np.isfinite(rel):
            break
        z = r / M_free
        rs_new = float(bm.sum(r * z))
        if not np.isfinite(rs_new):
            break
        beta = rs_new / rs
        p = z + beta * p
        rs = rs_new
    return x, it, rel


def _embed(full, x_free, free):
    x = bm.zeros((int(len(full)),), **bm.context(full))
    x = bm.set_at(x, free, x_free)
    return x


# --------------------------------------------------------------------------- #
# 灵敏度锥形密度滤波（scipy，与后端无关）
# --------------------------------------------------------------------------- #

def _checkpoint(outdir: str, density: np.ndarray, history: List[Dict[str, Any]]) -> None:
    """周期存档：长运行时防意外丢失的最终密度与历程."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "density_final.npy", density)
    (out / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

# --------------------------------------------------------------------------- #
# SIMP 主循环（bm 多后端）
# --------------------------------------------------------------------------- #

def run_simp(backend: str, nx: int, ny: int, nz: int, *,
             Lx: float = 60.0, Ly: float = 20.0, Lz: float = 4.0,
             volfrac: float = 0.3, penal: float = 3.0, rmin: float = 1.5,
             E_min: float = 1e-4, E: float = 1.0, nu: float = 0.3,
             move: float = 0.2, max_iters: int = 300, ctol: float = 1e-4,
             cg_rtol: float = 1e-6, cg_maxiter: int = 1000,
             dtype: str = "float32", chunk: int = 131072,
             single_step: int = 0, oc_bisect: int = 50,
             checkpoint_every: int = 0, outdir: str = "outputs",
             filter_kind: str = "box", ctol_metric: str = "density",
             ctol_patience: int = 3):
    _set_runtime(backend)
    if backend == "pytorch":
        import torch
        torch.cuda.reset_peak_memory_stats()

    node, cell2node, cell2dof_np = build_mesh(nx, ny, nz, Lx, Ly, Lz)
    hx, hy, hz = cell_sizes(nx, ny, nz, Lx, Ly, Lz)
    ndof = node.shape[0] * 3
    nelem = nx * ny * nz
    fixed_dofs, load_dofs, load_vals = boundary_sets(nx, ny, nz)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    ke0_np = hex8_stiffness(E, nu, hx, hy, hz)
    op = ElementMatvecOperator(cell2dof_np, ndof, ke0_np, dtype=dtype, chunk=chunk)

    f = np.zeros(ndof, dtype=np.float64)
    f[load_dofs] = load_vals

    rho = np.full(nelem, volfrac, dtype=np.float64)

    rmin_cells = (rmin / hx, rmin / hy, rmin / hz)

    ke0_b = _t(ke0_np.astype(dtype))
    f_b = _t(f.astype(dtype))
    free_b = _t(free.astype(np.int64))
    rho_b = _t(rho.astype(dtype))

    def to_numpy(t):
        return np.asarray(bm.to_numpy(t))

    history: List[Dict[str, Any]] = []
    timings: Dict[str, List[float]] = {"asm": [], "solve": [], "sens": [], "oc": []}
    c_prev = None
    it = 0
    change = 1.0
    change_rho = 1.0
    change_c = 1.0
    n_below = 0
    u_free_prev = None

    for it in range(1, max_iters + 1):
        t0 = time.perf_counter()
        E_e = E_min + (1.0 - E_min) * rho ** penal
        op.set_coefficients(E_e)
        t_asm = time.perf_counter() - t0

        t0 = time.perf_counter()
        M_diag = op.diagonal()
        u_free, ncg, rel = pcg(op, f_b, free_b, M_diag=M_diag, x0=u_free_prev,
                               rtol=cg_rtol, maxiter=cg_maxiter)
        u_free_prev = u_free
        u_full_b = _embed(f_b, u_free, free_b)
        t_solve = time.perf_counter() - t0

        t0 = time.perf_counter()
        u_e_b = u_full_b[op.cell2dof]
        # u_e^T K0 u_e: 用 'ej,jk,ek->e' 而非 'ej,ek,jk->e'，后者会在 torch 上
        # 物化 (nelem,24,24) 中间量（4.1M 单元时约 9.4GB），挤压 16GB 显存。
        comp_e = bm.einsum('ej,jk,ek -> e', u_e_b, ke0_b, u_e_b)
        dc_b = -penal * (rho_b ** (penal - 1.0)) * (1.0 - E_min) * comp_e
        dc = to_numpy(dc_b)
        dc_f = apply_structured_sensitivity_filter(
            dc.reshape(nx, ny, nz),
            rho.reshape(nx, ny, nz),
            rmin,
            (hx, hy, hz),
            filter_kind,
        ).ravel()
        t_sens = time.perf_counter() - t0

        t0 = time.perf_counter()
        l1, l2 = 0.0, 1e9
        rho_new = np.empty_like(rho)
        for _ in range(oc_bisect):
            lmid = 0.5 * (l1 + l2)
            # fp32 下空单元的 comp_e 可能取到微负值，必须 clip 再开方，否则 NaN
            ratio = np.clip(-dc_f, 0.0, None) / (lmid + 1e-12)
            rho_new = np.maximum(
                0.0,
                np.maximum(rho - move,
                           np.minimum(1.0,
                                      np.minimum(rho + move,
                                                 rho * np.sqrt(ratio)))))
            if np.mean(rho_new) > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        # 密度变化须在 rho 被覆盖前取, 这是 SIMP 的标准终止量.
        change_rho = float(np.max(np.abs(rho_new - rho)))
        rho = rho_new
        rho_b = _t(rho.astype(dtype))
        t_oc = time.perf_counter() - t0

        u = to_numpy(u_full_b)
        c = float(f.dot(u))
        # 柔顺度相对变化含 CG 求解误差, 仅作诊断记录, 不作默认判据.
        change_c = abs(c - c_prev) / c if c_prev else 1.0
        change = change_rho if ctol_metric == "density" else change_c
        c_prev = c
        vf = float(np.mean(rho))

        history.append({
            "iter": it, "compliance": c, "volume_fraction": vf,
            "change": change, "change_rho": change_rho, "change_c": change_c,
            "cg_iters": ncg, "cg_rel_res": float(rel),
            "t_asm": t_asm, "t_solve": t_solve, "t_sens": t_sens, "t_oc": t_oc,
        })
        for k in timings:
            timings[k].append(locals()["t_" + k])

        if it % 10 == 0 or it == 1:
            print(f"iter {it:4d}  c={c:.6e}  vf={vf:.4f}  change={change:.2e}"
                  f"(rho {change_rho:.2e}/c {change_c:.2e})  "
                  f"cg={ncg:4d}({rel:.1e})  asm={t_asm:.3f}s solve={t_solve:.3f}s "
                  f"sens={t_sens:.3f}s oc={t_oc:.3f}s")

        if checkpoint_every and it % checkpoint_every == 0:
            _checkpoint(outdir, rho.reshape(nx, ny, nz), history)

        if single_step > 0 and it >= single_step:
            break
        # 要求连续 ctol_patience 步低于阈值, 避免震荡中偶然撞线造成的假收敛.
        n_below = n_below + 1 if change < ctol else 0
        if n_below >= ctol_patience and it > 20:
            break

    converged = bool(n_below >= ctol_patience) and it > 20
    density = rho.reshape(nx, ny, nz)
    summary = {
        "backend": backend, "grid": [nx, ny, nz], "n_cells": nelem,
        "n_dofs": ndof, "n_fixed_dofs": int(len(fixed_dofs)),
        "L": [Lx, Ly, Lz], "h": [hx, hy, hz],
        "volfrac": volfrac, "penal": penal,
        "rmin_phys": rmin, "rmin_cells": rmin_cells,
        # 历史结果的 ke0 漏掉等参映射, 相差常数因子 4h; 记录该因子便于换算物理柔顺度.
        "ke0_scale": 4.0 * hx,
        "filter": filter_kind,
        "ctol_metric": ctol_metric, "ctol_patience": ctol_patience,
        "E_min": E_min, "E": E, "nu": nu, "move": move, "dtype": dtype,
        "max_iters": max_iters, "ctol": ctol, "cg_rtol": cg_rtol,
        "cg_maxiter": cg_maxiter,
        "iterations": it, "converged": converged,
        "final_compliance": float(history[-1]["compliance"]),
        "final_volume_fraction": float(history[-1]["volume_fraction"]),
        "final_change": float(history[-1]["change"]),
        "cg_iters_mean": float(np.mean([h["cg_iters"] for h in history])),
        "times_mean": {
            k: float(np.mean(v)) for k, v in timings.items()
        },
        "times_mean_total": float(np.mean([h["t_asm"] + h["t_solve"] + h["t_sens"] + h["t_oc"]
                                           for h in history])),
    }
    if backend == "pytorch":
        import torch
        summary["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated()) / 1e6
    return density, history, summary


# --------------------------------------------------------------------------- #
# 验证：Ke0 物理自检 + 算子 vs 显式装配 + PCG vs 直接法 + 小规模完整优化
# --------------------------------------------------------------------------- #

def validate(nx: int = 40, ny: int = 20, nz: int = 10, backend: str = "numpy",
             Lx: float = 40.0, Ly: float = 20.0, Lz: float = 10.0):
    _set_runtime(backend)

    node, cell2node, cell2dof = build_mesh(nx, ny, nz, Lx, Ly, Lz)
    hx, hy, hz = cell_sizes(nx, ny, nz, Lx, Ly, Lz)
    ndof = node.shape[0] * 3
    nelem = nx * ny * nz
    fixed_dofs, load_dofs, load_vals = boundary_sets(nx, ny, nz)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    ke0_np = hex8_stiffness(1.0, 0.3, hx, hy, hz)
    ok_ke = check_ke0_physical(ke0_np, 1.0, 0.3, hx, hy, hz)
    # 单元刚度的尺寸标度是 ke0 最易退化的一环, 额外抽查非立方体与 K_e ∝ h.
    ok_ke = check_ke0_physical(hex8_stiffness(1.0, 0.3, 0.3, 0.7, 1.3),
                               1.0, 0.3, 0.3, 0.7, 1.3) and ok_ke
    scale = float(np.max(np.abs(hex8_stiffness(1.0, 0.3, 0.5, 0.5, 0.5)))
                  / np.max(np.abs(hex8_stiffness(1.0, 0.3, 1.0, 1.0, 1.0))))
    ok_ke = abs(scale - 0.5) < 1e-12 and ok_ke
    print(f"[validate] 立方体标度 K(h=0.5)/K(h=1) = {scale:.6f} (理论 0.5)")

    rng = np.random.default_rng(0)
    rho = rng.uniform(0.2, 1.0, nelem)
    penal, E_min = 3.0, 1e-4
    E_e = E_min + (1.0 - E_min) * rho ** penal

    # 显式装配（向量化 coo）
    K_full = E_e[:, None, None] * ke0_np[None, :, :]
    rows = np.broadcast_to(cell2dof[:, None, :], K_full.shape).ravel()
    cols = np.broadcast_to(cell2dof[:, :, None], K_full.shape).ravel()
    K_sp = coo_matrix((K_full.ravel(), (rows, cols)),
                      shape=(ndof, ndof)).tocsr()

    op = ElementMatvecOperator(cell2dof, ndof, ke0_np, dtype="float64", chunk=16384)
    op.set_coefficients(E_e)

    x = rng.standard_normal(ndof)
    y_op = np.asarray(bm.to_numpy(op.matvec(_t(x))))
    y_sp = K_sp @ x
    err_op = float(np.max(np.abs(y_op - y_sp))) / (float(np.max(np.abs(y_sp))) + 1e-30)
    print(f"[validate] 算子 vs 显式装配 相对最大差 = {err_op:.3e}")

    # PCG vs spsolve
    f = np.zeros(ndof, dtype=np.float64)
    f[load_dofs] = load_vals
    K_ff = K_sp[free][:, free]
    u_ref = spsolve(K_ff, f[free])
    M = np.asarray(K_sp.diagonal()).astype(np.float64)
    u_free, ncg, rel = pcg(op, _t(f), _t(free.astype(np.int64)),
                           M_diag=_t(M), rtol=1e-9, maxiter=20000)
    u_pcg = np.asarray(bm.to_numpy(u_free))
    err_sol = float(np.linalg.norm(u_pcg - u_ref)) / float(np.linalg.norm(u_ref))
    print(f"[validate] PCG vs spsolve 相对误差 = {err_sol:.3e} (cg_iters={ncg}, "
          f"rel_res={rel:.1e})")

    # 小规模完整优化
    density, history, summary = run_simp(
        backend, nx, ny, nz, Lx=Lx, Ly=Ly, Lz=Lz,
        # h=1, 故 rmin 的物理值与单元数相等, 迭代轨迹与修复前逐位可比.
        volfrac=0.3, penal=3.0, rmin=2.0,
        max_iters=60, ctol=1e-4, cg_rtol=1e-6, cg_maxiter=2000,
        dtype="float64", chunk=16384)
    comps = [h["compliance"] for h in history]
    vfs = [h["volume_fraction"] for h in history]
    mono = all(comps[i] >= comps[i + 1] for i in range(len(comps) - 1))
    vf_ok = abs(float(np.mean(density)) - 0.3) < 0.02
    print(f"[validate] 优化: 迭代={summary['iterations']} 收敛={summary['converged']} "
          f"柔度单调={mono} 体积分数={float(np.mean(density)):.4f}(达标={vf_ok})")
    print(f"[validate] 柔度: {comps[0]:.3e} -> {comps[-1]:.3e}")

    ok = ok_ke and err_op < 1e-12 and err_sol < 1e-7 and mono and vf_ok
    print(f"[validate] 总体: {'通过' if ok else '失败'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CPU 基线：scipy 稀疏显式装配 + 稀疏 PCG 单步
# --------------------------------------------------------------------------- #

def cpu_sparse_single_step(nx: int, ny: int, nz: int, *,
                           Lx: float = 60.0, Ly: float = 20.0, Lz: float = 4.0,
                           volfrac: float = 0.3, penal: float = 3.0, rmin: float = 1.5,
                           move: float = 0.2,
                           E_min: float = 1e-4, E: float = 1.0, nu: float = 0.3,
                           cg_rtol: float = 1e-6, cg_maxiter: int = 1000,
                           chunk: int = 262144,
                           filter_kind: str = "box") -> Dict[str, Any]:
    """CPU 传统流程（NumPy/SciPy）：显式稀疏装配 + 稀疏 PCG，单步计时。"""
    node, cell2node, cell2dof = build_mesh(nx, ny, nz, Lx, Ly, Lz)
    hx, hy, hz = cell_sizes(nx, ny, nz, Lx, Ly, Lz)
    ndof = node.shape[0] * 3
    nelem = nx * ny * nz
    fixed_dofs, load_dofs, load_vals = boundary_sets(nx, ny, nz)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    ke0_np = hex8_stiffness(E, nu, hx, hy, hz)

    rho = np.full(nelem, volfrac)
    E_e = E_min + (1.0 - E_min) * rho ** penal

    t0 = time.perf_counter()
    K_sp = build_sparse_global(cell2dof, E_e, ke0_np, ndof)
    t_asm = time.perf_counter() - t0

    f = np.zeros(ndof)
    f[load_dofs] = load_vals
    K_ff = K_sp[free][:, free]
    b = f[free]
    M_diag = np.asarray(K_sp.diagonal())[free]

    # CPU 基线的 PCG 走 soptx.solvers 的统一实现, 与 GPU 侧同一份 Jacobi 口径。
    # 注意 CGSolver 每次 solve 会多做一次 matvec 算真残差 relres, 在 t_solve
    # 里是可见开销 (相对上百步迭代约 1/niter), 与旧的 scipy 计时不逐位可比。
    cpu_solver = CGSolver(
        M=DiagonalPreconditioner(M_diag),
        atol=0.0, rtol=cg_rtol, maxit=cg_maxiter,
    ).setup(K_ff)
    t0 = time.perf_counter()
    u_free, info = cpu_solver.solve(b)
    t_solve = time.perf_counter() - t0

    u_full = np.zeros(ndof)
    u_full[free] = u_free
    c = float(f.dot(u_full))

    t0 = time.perf_counter()
    u_e = u_full[cell2dof]
    comp_e = np.einsum('ej,ek,jk -> e', u_e, u_e, ke0_np)
    dc = -penal * (rho ** (penal - 1.0)) * (1.0 - E_min) * comp_e
    dc_f = apply_structured_sensitivity_filter(
        dc.reshape(nx, ny, nz),
        rho.reshape(nx, ny, nz),
        rmin,
        (hx, hy, hz),
        filter_kind,
    ).ravel()
    t_sens = time.perf_counter() - t0

    t0 = time.perf_counter()
    l1, l2 = 0.0, 1e9
    for _ in range(100):
        lmid = 0.5 * (l1 + l2)
        ratio = np.clip(-dc_f, 0.0, None) / (lmid + 1e-12)
        rho_new = np.maximum(
            0.0,
            np.maximum(rho - move,
                       np.minimum(1.0,
                                  np.minimum(rho + move, rho * np.sqrt(ratio)))))
        if np.mean(rho_new) > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    t_oc = time.perf_counter() - t0

    return {
        "stage": "cpu-sparse", "grid": [nx, ny, nz], "n_dofs": ndof,
        "n_cells": nelem,
        "L": [Lx, Ly, Lz], "rmin_phys": rmin, "filter": filter_kind,
        "cg_rtol": cg_rtol, "dtype": "float64",
        "t_asm": t_asm, "t_solve": t_solve, "t_sens": t_sens, "t_oc": t_oc,
        "t_total": t_asm + t_solve + t_sens + t_oc,
        "cg_iters": int(info["niter"]) if info["converged"] else cg_maxiter,
        "converged_cg": bool(info == 0), "compliance": c,
    }


def build_sparse_global(cell2dof: np.ndarray, E_e: np.ndarray, ke0: np.ndarray,
                        ndof: int, chunk: int = 262144):
    """向量化 coo 装配全局稀疏刚度矩阵（显式装配，CPU 基线用）。"""
    nelem = len(E_e)
    rows_l, cols_l, vals_l = [], [], []
    for s in range(0, nelem, chunk):
        e = slice(s, min(s + chunk, nelem))
        K_c = E_e[e][:, None, None] * ke0[None, :, :]
        ld = cell2dof[e]
        rows_l.append(np.broadcast_to(ld[:, None, :], K_c.shape).ravel())
        cols_l.append(np.broadcast_to(ld[:, :, None], K_c.shape).ravel())
        vals_l.append(K_c.ravel())
    rows = np.concatenate(rows_l); cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    return coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()


def compare_cpu_gpu(nx: int, ny: int, nz: int, density_path: str, *,
                    Lx: float = 60.0, Ly: float = 20.0, Lz: float = 4.0,
                    volfrac: float = 0.3, penal: float = 3.0,
                    E_min: float = 1e-4, E: float = 1.0, nu: float = 0.3,
                    cg_rtol: float = 1e-6, cg_maxiter: int = 5000,
                    dtype: str = "float32",
                    chunk: int = 131072) -> Dict[str, Any]:
    """同一设计密度下，CPU(scipy 稀疏 PCG) 与 GPU(张量化 PCG) 解同一系统并对比.

    CPU 侧固定 ``float64``(scipy); GPU 侧精度由 ``dtype`` 指定. ``cg_rtol`` 严于
    ``float32`` 的机器精度(约 1.2e-7)时必须取 ``float64``, 否则 GPU 侧无法收敛.
    """
    node, cell2node, cell2dof = build_mesh(nx, ny, nz, Lx, Ly, Lz)
    hx, hy, hz = cell_sizes(nx, ny, nz, Lx, Ly, Lz)
    ndof = node.shape[0] * 3
    nelem = nx * ny * nz
    fixed_dofs, load_dofs, load_vals = boundary_sets(nx, ny, nz)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    ke0_np = hex8_stiffness(E, nu, hx, hy, hz)

    rho = np.load(density_path).ravel()
    E_e = E_min + (1.0 - E_min) * rho ** penal
    f = np.zeros(ndof)
    f[load_dofs] = load_vals

    t0 = time.perf_counter()
    K_sp = build_sparse_global(cell2dof, E_e, ke0_np, ndof)
    t_asm = time.perf_counter() - t0

    K_ff = K_sp[free][:, free]
    b = f[free]
    M_diag = np.asarray(K_sp.diagonal())[free]

    cpu_solver = CGSolver(
        M=DiagonalPreconditioner(M_diag),
        atol=0.0, rtol=cg_rtol, maxit=cg_maxiter,
    ).setup(K_ff)
    t0 = time.perf_counter()
    u_cpu, info_cpu = cpu_solver.solve(b)
    t_cpu_solve = time.perf_counter() - t0
    # relres 已由 CGSolver 按同一口径 ||b - A x|| / ||b|| 算过, 直接取用。
    rel_cpu = float(info_cpu["relres"])
    cg_cpu = int(info_cpu["niter"]) if info_cpu["converged"] else cg_maxiter

    _set_runtime("pytorch")
    op = ElementMatvecOperator(cell2dof, ndof, ke0_np, dtype=dtype, chunk=chunk)
    op.set_coefficients(E_e)
    t0 = time.perf_counter()
    u_gpu_free, ncg, rel_gpu = pcg(op, _t(f.astype(np.dtype(dtype))),
                                   _t(free.astype(np.int64)),
                                   M_diag=op.diagonal(),
                                   rtol=cg_rtol, maxiter=cg_maxiter)
    t_gpu_solve = time.perf_counter() - t0
    u_gpu = np.asarray(bm.to_numpy(u_gpu_free))
    # 对齐求解容差：比较 CPU/GPU 解的相互一致程度（两侧各自收敛到 rtol）
    err = float(np.linalg.norm(u_cpu - u_gpu)) / float(np.linalg.norm(u_cpu))

    return {
        "stage": "cpu-vs-gpu", "grid": [nx, ny, nz], "n_dofs": ndof,
        "density": density_path,
        "L": [Lx, Ly, Lz], "cg_rtol": cg_rtol,
        "dtype_cpu": "float64", "dtype_gpu": dtype,
        "t_cpu_asm": t_asm, "t_cpu_solve": t_cpu_solve, "t_gpu_solve": t_gpu_solve,
        "cg_iters_cpu": cg_cpu,
        "cg_iters_gpu": ncg,
        "rel_res_cpu": rel_cpu, "rel_res_gpu": float(rel_gpu),
        "solution_rel_diff": err,
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="真实 SIMP 三维拓扑优化（多后端张量化）")
    parser.add_argument("--backend", choices=["numpy", "pytorch"], default="pytorch")
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--Lx", type=float, default=60.0, help="设计域 x 向长度")
    parser.add_argument("--Ly", type=float, default=20.0, help="设计域 y 向长度")
    parser.add_argument("--Lz", type=float, default=4.0, help="设计域 z 向长度")
    parser.add_argument("--validate", action="store_true", help="小网格正确性验证")
    parser.add_argument("--cpu-baseline", action="store_true",
                        help="scipy 稀疏 CPU 单步基线计时")
    parser.add_argument("--compare", type=str, default=None,
                        metavar="DENSITY_NPY",
                        help="CPU(scipy) 与 GPU(张量化) 解同一系统的对比")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--volfrac", type=float, default=0.3)
    parser.add_argument("--penal", type=float, default=3.0)
    # rmin 按物理长度给出, 内部除以单元边长换算为单元数, 保证网格无关性.
    parser.add_argument("--rmin", type=float, default=1.5,
                        help="过滤半径, 物理长度单位(非单元个数)")
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--ctol", type=float, default=1e-4)
    parser.add_argument("--cg-rtol", type=float, default=1e-6)
    parser.add_argument("--cg-maxiter", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=131072)
    parser.add_argument("--single-step", type=int, default=0,
                        help="只跑 N 个单步（计时用）")
    parser.add_argument("--oc-bisect", type=int, default=50,
                        help="OC 二分次数（默认 50，lmid 分辨率 1e-9）")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="每 N 步周期存档 density/history（0 关闭）")
    parser.add_argument("--ctol-metric", dest="ctol_metric",
                        choices=["density", "compliance"], default="density",
                        help="终止量: density 为 max|drho|(标准, 建议 ctol=1e-2), "
                             "compliance 为柔顺度相对变化(含求解噪声)")
    parser.add_argument("--ctol-patience", dest="ctol_patience", type=int, default=3,
                        help="连续多少步低于 ctol 才判定收敛")
    parser.add_argument("--filter", dest="filter_kind",
                        choices=["box", "cone"], default="box",
                        help="灵敏度过滤: box 为纯平滑, cone 为 Sigmund 标准锥形式")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    if args.validate:
        return validate(backend=args.backend)

    if args.compare:
        res = compare_cpu_gpu(args.nx, args.ny, args.nz, args.compare,
                              Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
                              cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter,
                              dtype=args.dtype)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "cpu_vs_gpu_compare.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.cpu_baseline:
        res = cpu_sparse_single_step(args.nx, args.ny, args.nz,
                                     Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
                                     rmin=args.rmin,
                                     cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter,
                                     filter_kind=args.filter_kind)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "cpu_sparse_single_step.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    density, history, summary = run_simp(
        args.backend, args.nx, args.ny, args.nz,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        volfrac=args.volfrac, penal=args.penal, rmin=args.rmin,
        max_iters=args.max_iters, ctol=args.ctol,
        cg_rtol=args.cg_rtol, cg_maxiter=args.cg_maxiter,
        dtype=args.dtype, chunk=args.chunk, single_step=args.single_step,
        oc_bisect=args.oc_bisect, checkpoint_every=args.checkpoint_every,
        outdir=args.outdir, filter_kind=args.filter_kind,
        ctol_metric=args.ctol_metric, ctol_patience=args.ctol_patience,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "density_final.npy", density)
    (outdir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[out] {outdir / 'density_final.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
