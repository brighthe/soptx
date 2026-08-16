"""PIML 代理缩聚与精确 Schur 补缩聚的对比验证入口.

本脚本在同一物理问题, 同一密度场与同一接口系统上比较两条缩聚路径: 精确
``FEAStaticCondensation`` 批量 Schur 补, 与 ``PIMLStaticCondensation`` 逐子结构
代理预测. 二者的差异分两层报告: 算子层给出 ``K_s`` 的相对 Frobenius 误差, 是代理
自身的精度, 与外载无关; 解层给出接口位移, 全场位移与柔度的相对误差, 是算子误差经
接口系统放大后的后果. 两层之比反映误差在求解链路上的传播倍率.

精确缩聚在此作为基线而非被验对象: 它与 Lagrange 全装配的机器精度等价已由
``examples/substructure_elasticity/compare_lagrange.py`` 建立, 本脚本不再重复验证,
因此不构造全尺度参考解. 物理问题取 ``FullMBBBeam2d`` 完整 MBB 梁, 对齐 Huang 2023
第 4.1 节.

解层误差远大于算子层误差时, 仅凭这两层无法判断责任在网络还是在参数化, 因此脚本还
输出一组误差归因诊断: 训练同分布留出集与光滑评估场的误差之比区分欠拟合与分布错配;
代理在精确 ``K_s`` 刚体零空间上注入的伪刚度, 以及该伪刚度在精确解上贡献的应变能占比,
度量软模态是否被污染. 该诊断曾定位出主导误差源: 历史上的 ``L L^T`` 参数化结构上无法
表示秩亏, 而接口位移几乎全部是刚体运动, 使得零空间上很小的伪刚度被平方量级放大.
现参数化把预测限制到刚体模态的正交补上, 该通道由构造关闭, 诊断转为对该构造的运行期
校验. 详见 ``results_analysis.md``.

代理精度没有机器精度阈值可断言, 因此默认只报告不验收; ``--strict`` 下若在役子结构或
留出样本上存在回退到精确缩聚的情形则以异常失败, 使"代理是否真正投入使用"成为可回归
的状态.

使用方法:
    # 默认配置: 300 组训练样本, 400 轮训练, 100 组留出样本.
    python examples/piml_substructure_elasticity/compare_exact.py

    # 要求代理全程生效, 出现回退即失败.
    python examples/piml_substructure_elasticity/compare_exact.py --strict --epochs 800

训练与采样的随机性由 ``--seed`` 统一固定, 同一组参数的两次运行逐位可复现; 比较不同
训练配置时应保持种子不变, 否则观测到的差异混有采样噪声.

``--output-dir`` 缺省为本脚本同级的 ``outputs/``, 按脚本位置解析, 与从哪个目录发起
命令无关; 传相对路径时按当前工作目录解析, 可能落到 ``.gitignore`` 覆盖范围之外.
"""

import json
import time
import argparse
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm

from soptx.fem.solvers import LagrangeFEMAnalyzer
from soptx.problems.elasticity import FullMBBBeam2d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.fem.substructure import (
    FEAStaticCondensation,
    GlobalAssembler,
    PIMLStaticCondensation,
    PIMLSurrogateNet,
    SubstructureMesh,
    SubstructurePrototype,
    solve_interface_system,
)


# 训练样本的密度取值区间; 评估密度场必须落在其中, 否则代理工作在外插区间.
DENSITY_RANGE = (0.3, 1.0)

# 自由漂浮子结构的 K_s 以刚体模态为精确零空间, 只是半正定. 参数化把它限制到刚体
# 模态的正交补上再取 Cholesky 因子, 限制后严格正定, 因此不再需要任何正则项.

TABLE_WIDTHS = (38, 22, 22)

# 诊断段落为标签—数值两列, 与三列结果表用不同的对齐宽度.
DIAG_LABEL_WIDTH = 46


### 表格排版 ###

def display_width(s: str) -> int:
    """计算字符串在等宽终端下的显示宽度, 东亚全角字符按两列计."""
    width = 0
    for char in s:
        width += 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
    return width


def format_table_row(col1: str, col2: str, col3: str) -> str:
    """按显示宽度对齐三列表格行."""
    return (
        f"{col1}{' ' * (TABLE_WIDTHS[0] - display_width(col1))} | "
        f"{col2}{' ' * (TABLE_WIDTHS[1] - display_width(col2))} | "
        f"{col3}{' ' * (TABLE_WIDTHS[2] - display_width(col3))}"
    )


def format_diag_row(label: str, value: str) -> str:
    """按显示宽度对齐诊断段落的标签与数值两列."""
    pad = max(1, DIAG_LABEL_WIDTH - display_width(label))
    return f"{label}{' ' * pad} : {value}"


### 随机性 ###

def set_random_seed(seed: int) -> None:
    """统一固定 ``bm`` 后端与 PyTorch 的随机数种子.

    参数:
        seed: 随机数种子.

    说明:
        训练样本采样, 留出集采样与网络参数初始化是本脚本仅有的三处随机性来源, 全部
        由这两个种子覆盖. 必须在 ``bm.set_backend`` 之后调用: ``bm.random`` 的实现由
        当前后端决定.
    """
    bm.random.seed(seed)
    torch.manual_seed(seed)


### 子结构构造与密度场 ###

def build_substructures(
    assembler: GlobalAssembler,
) -> Tuple[SubstructurePrototype, List[SubstructureMesh], List[Tuple[int, ...]]]:
    """按装配器的布局铺开全部子结构, 共享同一个参考子结构.

    参数:
        assembler: 已构造的全局装配器, 提供求解域尺寸与子结构划分.

    返回:
        (prototype, sub_meshes, positions): 共享的参考子结构, 按 x 优先字典序排列的
            子结构列表, 以及各子结构在子结构网格中的整数位置 ``(sx, sy)``. 位置与
            ``sub_meshes`` 同序, 供 ``get_substructure_global_dofs`` 把局部自由度映射
            到全局编号. 全部子结构同构, 因此离散结构, 自由度划分与单位密度单元刚度
            只构造一次.
    """
    sub_size = tuple(
        assembler.domain_size[d] / assembler.n_sub[d] for d in range(assembler.dim)
    )
    prototype = SubstructurePrototype(
        sub_size, assembler.n_fine, assembler.E_base, assembler.nu
    )

    sub_meshes: List[SubstructureMesh] = []
    positions: List[Tuple[int, ...]] = []
    sub_id = 0
    for sx in range(assembler.n_sub[0]):
        for sy in range(assembler.n_sub[1]):
            spans = (
                (sx * sub_size[0], (sx + 1) * sub_size[0]),
                (sy * sub_size[1], (sy + 1) * sub_size[1]),
            )
            sub_meshes.append(
                SubstructureMesh(
                    sub_id, *spans, *assembler.n_fine,
                    assembler.E_base, assembler.nu, prototype=prototype,
                )
            )
            positions.append((sx, sy))
            sub_id += 1
    return prototype, sub_meshes, positions


def make_density_fields(
    sub_meshes: Sequence[SubstructureMesh],
    domain_size: Sequence[float],
) -> Any:
    """生成逐细单元变化的批量局部密度场.

    参数:
        sub_meshes: 子结构列表, 全部同构.
        domain_size: 求解域在各方向的尺寸 ``(Lx, Ly)``.

    返回:
        density: 形状 ``(B, nx, ny)`` 的批量密度场, 前导维 ``B`` 与 ``sub_meshes``
            同序, 后两维为子结构细网格的单元排布.

    说明:
        密度在子结构**内部**也随单元中心坐标变化, 而非每个子结构取一个常值. 代理网络
        的训练输入是各分量独立的随机密度, 若评估时喂入分量全相等的均匀密度, 只能测到
        训练分布中一个测度为零的切片. 幅值取在 ``DENSITY_RANGE`` 内, 保证评估工作在
        内插而非外插区间.
    """
    n_fine = tuple(sub_meshes[0].n_fine)
    dim = len(domain_size)

    # (B, dim) 的子结构包围盒下界与跨度, 用于把单位区间上的单元中心映射到物理坐标.
    lows = bm.asarray(
        [[sm.box_span[d][0] for d in range(dim)] for sm in sub_meshes],
        dtype=bm.float64,
    )
    spans = bm.asarray(
        [[sm.box_span[d][1] - sm.box_span[d][0] for d in range(dim)] for sm in sub_meshes],
        dtype=bm.float64,
    )

    # 单元中心在子结构局部单位区间上的位置, 形状分别为 (nx,) 与 (ny,).
    unit_x = (bm.arange(n_fine[0], dtype=bm.float64) + 0.5) / n_fine[0]
    unit_y = (bm.arange(n_fine[1], dtype=bm.float64) + 0.5) / n_fine[1]

    # (B, nx, 1) 与 (B, 1, ny) 相乘时广播为 (B, nx, ny), 无需显式 meshgrid.
    x = (lows[:, 0:1] + unit_x[None, :] * spans[:, 0:1])[:, :, None]
    y = (lows[:, 1:2] + unit_y[None, :] * spans[:, 1:2])[:, None, :]

    lo, hi = DENSITY_RANGE
    mid = 0.5 * (lo + hi)
    # 留出 10% 余量, 使密度严格落在训练区间内部而非贴边.
    amp = 0.45 * (hi - lo)
    modulation = bm.sin(bm.pi * x / domain_size[0]) * bm.cos(bm.pi * y / domain_size[1])
    return mid + amp * modulation


def sample_random_density(prototype: SubstructurePrototype, n_sample: int) -> Any:
    """按训练分布采样一批随机局部密度.

    参数:
        prototype: 共享参考子结构, 提供细网格规模.
        n_sample: 采样组数.

    返回:
        rho: 形状 ``(n_sample, nx, ny)`` 的密度, 各分量在 ``DENSITY_RANGE`` 上独立
            均匀采样.

    说明:
        训练集与留出集共用本函数, 以此保证两者严格同分布——留出误差与光滑评估场误差
        的对比只有在这一前提下才能归因于分布错配.
    """
    # 后端在运行时接受多个尺寸参数, 但静态类型声明仅暴露单个参数.
    random_rand = cast(Callable[..., Any], bm.random.rand)
    lo, hi = DENSITY_RANGE
    return lo + (hi - lo) * bm.asarray(
        random_rand(n_sample, *tuple(prototype.n_fine)), dtype=bm.float64
    )


### 代理网络训练 ###

def _to_torch_training_tensor(values: List[Any]) -> torch.Tensor:
    """将后端数据适配为 PyTorch 训练张量.

    参数:
        values: 一批同形状的后端数组.

    返回:
        tensor: 堆叠后的 ``float32`` 训练张量.

    说明:
        PyTorch 后端直接复用 ``bm.stack`` 的张量; 其他后端经 ``bm.to_numpy`` 规范
        转换, 使有限元数据生成与缩聚过程不依赖特定后端.
    """
    stacked = bm.stack(values)
    if isinstance(stacked, torch.Tensor):
        return stacked.to(dtype=torch.float32)
    return torch.from_numpy(bm.to_numpy(stacked)).to(dtype=torch.float32)


def train_surrogate(
    prototype: SubstructurePrototype,
    n_train: int,
    n_epochs: int,
    learning_rate: float,
) -> Tuple[PIMLSurrogateNet, float]:
    """在随机密度样本上训练 Cholesky 因子代理网络.

    参数:
        prototype: 共享参考子结构, 提供细网格规模与自由度划分.
        n_train: 随机密度训练样本数.
        n_epochs: 全批量梯度下降的迭代轮数.
        learning_rate: Adam 学习率.

    返回:
        (net, final_loss): 训练完毕并置于 ``eval`` 模式的网络, 以及最后一轮的
            训练 MSE.

    说明:
        训练集一次批量生成: ``n_train`` 组随机密度共用同一套离散结构, 局部刚度装配
        与 Schur 补缩聚各只调用一次. 拟合目标是变形子空间上的无正则 Cholesky 因子
        ``cholesky(R^T K_s R)`` 的下三角独立条目, ``R`` 为刚体模态的正交补. 限制后
        的算子严格正定, 无需正则项, 因此目标不带正偏置; 推理侧按
        ``R L L^T R^T`` 重构, 刚体零空间由构造精确保持. 详见
        ``PIMLStaticCondensation`` 的类说明.
    """
    n_fine = tuple(prototype.n_fine)
    basis = prototype.deformation_basis
    n_reduced = int(basis.shape[1])
    tril_mask = bm.tril(bm.ones((n_reduced, n_reduced), dtype=bm.bool))
    n_tril = int(bm.sum(tril_mask))

    rand_rho = sample_random_density(prototype, n_train)

    K_train_batch = prototype.assemble_local_stiffness_batch(rand_rho)
    train_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    K_s_train, _ = train_condensor.condense(K_train_batch)

    # 限制到变形子空间后算子严格正定, Cholesky 分解无需任何正则.
    K_s_reduced = basis.T @ K_s_train @ basis
    L_train = bm.linalg.cholesky(K_s_reduced)

    X_train = _to_torch_training_tensor(
        [bm.reshape(rand_rho[i], (-1,)) for i in range(n_train)]
    )
    Y_train = _to_torch_training_tensor(
        [L_train[i][tril_mask] for i in range(n_train)]
    )

    net = PIMLSurrogateNet(input_dim=n_fine[0] * n_fine[1], output_dim=n_tril)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    net.train()
    final_loss = float("nan")
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train), Y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    net.eval()
    return net, final_loss


### 接口系统求解 ###

def solve_with_condensors(
    assembler: GlobalAssembler,
    sub_meshes: List[SubstructureMesh],
    condensors: Any,
    global_load: Any,
    fixed_global_dofs: Any,
) -> Tuple[Any, Any, int]:
    """用给定的缩聚结果装配并求解全局接口系统, 再恢复全场位移.

    参数:
        assembler: 全局装配器.
        sub_meshes: 子结构列表.
        condensors: 单个批量缩聚器, 或与 ``sub_meshes`` 同序的缩聚器列表.
        global_load: 施加 Dirichlet 条件之前的全局外载向量, 形状 ``(n_full,)``.
        fixed_global_dofs: 固定自由度的全局编号.

    返回:
        (u_interface, u_full, n_free): 接口位移, 恢复后的全场位移, 以及施加约束后
            接口系统的求解自由度数.

    说明:
        两条路径共用同一批 ``sub_meshes``, 因此接口自由度编号一致, 接口位移可逐分量
        直接相减.
    """
    system = assembler.assemble_interface_system(sub_meshes, condensors)
    interface_fixed = assembler.project_global_dofs(system, fixed_global_dofs)
    u_interface = solve_interface_system(
        system,
        assembler.project_global_vector(system, global_load),
        interface_fixed,
    )
    u_full = assembler.recover_full_displacement(
        sub_meshes, condensors, system, u_interface
    )
    n_free = int(len(system.global_dofs)) - int(len(interface_fixed))
    return u_interface, u_full, n_free


### 误差归因诊断 ###

class _ExactCholeskyStub(nn.Module):
    """恒定输出给定 Cholesky 条目的桩网络, 用于模拟"完美代理"."""

    def __init__(self, entries: Any) -> None:
        super().__init__()
        self.register_buffer(
            "entries",
            torch.as_tensor(bm.to_numpy(entries), dtype=torch.float32).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.entries)


def verify_parameterization_parity(
    prototype: SubstructurePrototype,
) -> Dict[str, Any]:
    """校验刚体模态基的正确性, 以及训练目标与推理重构是同一个参数化.

    参数:
        prototype: 共享参考子结构.

    返回:
        字典, 含以下键:
            ``rigid_basis_residual``: ``||K_s R_rigid|| / ||K_s||``, 检验解析构造的
                刚体模态基确实是精确 ``K_s`` 的零空间.
            ``parameterization_error_ceiling``: 完美代理下 ``K_s`` 的相对 Frobenius
                误差, 即该参数化本身的精度上限.

    异常:
        AssertionError: 刚体基残差超过 ``1e-10``, 桩网络触发回退, 或重构误差超过
            ``1e-4`` 时抛出.

    说明:
        本检查串联两件事. 其一, 解析构造的刚体模态基必须张成 ``K_s`` 的零空间, 否则
        把预测限制到它的正交补上会抹掉真实刚度; 该性质对 P1 拉格朗日单元精确成立,
        实测残差在机器精度量级. 其二, 训练侧用布尔掩码按行优先取 ``L`` 的下三角条目,
        推理侧用同一掩码写回; 两处排布若不一致, 网络会去拟合一个被置换过的目标, 训练
        损失照常下降而预测的 ``K_s`` 完全错误, 且不触发任何门禁.

        误差上限同时是本次运行的误差下界: 代理的 ``K_s`` 相对误差不可能低于它. 桩网络
        按 ``float32`` 存储条目, 与真实网络的输出精度一致, 因此上限反映的是参数化在
        单精度下的极限, 实测约 ``3e-8``.
    """
    basis = prototype.deformation_basis
    n_reduced = int(basis.shape[1])
    rho = sample_random_density(prototype, 1)
    K_local = prototype.assemble_local_stiffness_batch(rho)
    K_s_batch, _ = FEAStaticCondensation(
        prototype.i_dofs, prototype.b_dofs
    ).condense(K_local)
    K_s = K_s_batch[0]

    K_s_norm = float(bm.linalg.norm(K_s))
    rigid_residual = float(bm.linalg.norm(K_s @ prototype.rigid_basis)) / K_s_norm
    if rigid_residual > 1.0e-10:
        raise AssertionError(
            f"刚体模态基残差 ||K_s R||/||K_s|| = {rigid_residual:.3e} 过大, "
            f"该基未张成 K_s 的零空间, 限制到其正交补会丢失真实刚度"
        )

    L = bm.linalg.cholesky(basis.T @ K_s @ basis)
    tril_mask = bm.tril(bm.ones((n_reduced, n_reduced), dtype=bm.bool))

    condensor = PIMLStaticCondensation(
        prototype.i_dofs, prototype.b_dofs,
        model=_ExactCholeskyStub(L[tril_mask]), is_cholesky=True,
        range_basis=basis,
    )
    K_s_pred, _ = condensor.condense(K_local[0], rho[0])
    if condensor.used_fallback:
        raise AssertionError("完美 Cholesky 条目触发了回退, 推理路径与训练目标不一致")

    ceiling = float(bm.linalg.norm(K_s_pred - K_s)) / K_s_norm
    if ceiling > 1.0e-4:
        raise AssertionError(
            f"完美代理下 K_s 相对误差 {ceiling:.3e} 过大, "
            f"训练目标的条目排布与推理期的重构不一致"
        )
    return {
        "rigid_basis_residual": rigid_residual,
        "parameterization_error_ceiling": ceiling,
    }


def evaluate_holdout(
    prototype: SubstructurePrototype,
    net: PIMLSurrogateNet,
    n_val: int,
) -> Dict[str, Any]:
    """在与训练同分布的留出集上评估代理的算子层精度.

    参数:
        prototype: 共享参考子结构.
        net: 已训练完毕并置于 ``eval`` 模式的代理网络.
        n_val: 留出样本数.

    返回:
        stats: 含留出集 ``K_s`` 相对 Frobenius 误差的 max 与 mean, 以及留出集上的
            回退计数.

    说明:
        留出密度由 ``sample_random_density`` 采样, 与训练集严格同分布且以概率 1 不重合,
        因此本误差度量的是"网络在自己的训练分布上究竟学到了什么". 它与主流程在光滑密度
        场上的误差之比即分布错配的代价: 两者接近说明瓶颈是欠拟合, 留出显著更小说明瓶颈
        是训练分布不覆盖光滑输入.

        推理走与主流程同一条 ``PIMLStaticCondensation`` 路径, 因此对角线取绝对值与
        投影回全部接口自由度等推理期处理都被计入, 度量的是实际投入使用的算子而非网络
        的原始输出.
    """
    rho_val = sample_random_density(prototype, n_val)
    K_val_batch = prototype.assemble_local_stiffness_batch(rho_val)

    ref_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    K_s_ref, _ = ref_condensor.condense(K_val_batch)

    piml_condensor = PIMLStaticCondensation(
        prototype.i_dofs, prototype.b_dofs, model=net, is_cholesky=True,
        range_basis=prototype.deformation_basis,
    )
    K_s_pred_list: List[Any] = []
    n_fallback = 0
    for i in range(n_val):
        piml_condensor.condense(K_val_batch[i], rho_val[i])
        n_fallback += int(piml_condensor.used_fallback)
        K_s_pred_list.append(piml_condensor.K_s)
    K_s_pred = bm.stack(K_s_pred_list, axis=0)

    err = bm.linalg.norm(
        K_s_pred - K_s_ref, axis=(-2, -1)
    ) / bm.linalg.norm(K_s_ref, axis=(-2, -1))

    return {
        "holdout_samples": n_val,
        "holdout_ks_relative_error_max": float(bm.max(err)),
        "holdout_ks_relative_error_mean": float(bm.mean(err)),
        "holdout_n_fallback": n_fallback,
    }


def diagnose_rigid_mode_pollution(
    K_s_exact: Any,
    K_s_piml: Any,
    u_b_sub: Any,
    n_rigid: int,
) -> Dict[str, Any]:
    """量化代理在精确 ``K_s`` 零空间上注入的伪刚度及其能量后果.

    参数:
        K_s_exact: 精确缩聚刚度批量, 形状 ``(B, n_b, n_b)``.
        K_s_piml: 代理缩聚刚度批量, 形状 ``(B, n_b, n_b)``.
        u_b_sub: 精确解在各子结构接口自由度上的取值, 形状 ``(B, n_b)``, 列序与
            ``K_s_exact`` 的自由度序一致.
        n_rigid: 刚体模态数, 二维为 3, 三维为 6.

    返回:
        stats: 零空间特征量, 伪刚度与应变能归因的统计字典.

    说明:
        自由漂浮子结构的精确 ``K_s`` 恰有 ``n_rigid`` 个零特征值. 刚体模态是最软的
        方向, 也正是装配后各子结构位移的主要成分, 因此该子空间上的任何伪刚度, 其能量
        后果都可以远大于它在 Frobenius 范数下的占比. 历史上的 ``L L^T`` 参数化严格
        正定, 结构上无法表示秩亏, 曾在此注入约 ``1.6e-2`` 相对量级的伪刚度并造成十余倍
        的刚化; 现参数化把预测限制在刚体模态的正交补上, 该项应降到机器精度.

        本诊断刻意不复用 ``prototype.rigid_basis``: 零空间由精确 ``K_s`` 的最小
        ``n_rigid`` 个特征向量独立给出, 因此它同时校验解析构造的刚体基是否正确, 也不
        假设网格坐标, 二维与三维通用. ``rigid_pollution_ratio`` 以精确 ``K_s`` 的最小
        非零特征值为标尺, 是与载荷无关的算子层指标; ``energy_*`` 则在精确解上求值,
        给出解层的能量归因.
    """
    evals, evecs = bm.linalg.eigh(K_s_exact)
    # 特征值升序, 前 n_rigid 个对应刚体模态; V 的列已正交归一.
    V = evecs[..., :n_rigid]
    Vt = bm.swapaxes(V, -1, -2)

    lam_rigid = bm.max(bm.abs(evals[:, :n_rigid]), axis=-1)
    lam_soft = evals[:, n_rigid]

    proj_exact = Vt @ K_s_exact @ V
    proj_piml = Vt @ K_s_piml @ V
    # 精确算子在零空间上的残余取绝对值最大者: 该值应在机器精度量级, 作为 V 的自检.
    pol_exact = bm.max(bm.abs(bm.linalg.eigvalsh(proj_exact)), axis=-1)
    # 代理算子在零空间上的伪刚度. 现参数化下投影应为零阵, 残余可正可负, 故同样取
    # 绝对值最大的特征值, 度量的是偏离零的幅度而非某个符号方向.
    pol_piml = bm.max(bm.abs(bm.linalg.eigvalsh(proj_piml)), axis=-1)
    pol_ratio = pol_piml / lam_soft

    # 在精确解上求值: 两条路径的应变能之比即代理"看到"的刚化倍率.
    e_exact = bm.einsum('bi,bij,bj->b', u_b_sub, K_s_exact, u_b_sub)
    e_piml = bm.einsum('bi,bij,bj->b', u_b_sub, K_s_piml, u_b_sub)
    # 位移在刚体子空间上的坐标, 以及伪刚度在该子空间上单独贡献的能量.
    a = bm.einsum('bkr,bk->br', V, u_b_sub)
    e_rigid = bm.einsum('br,brs,bs->b', a, proj_piml, a)
    rigid_fraction = bm.linalg.norm(a, axis=-1) / bm.linalg.norm(u_b_sub, axis=-1)

    e_exact_total = float(bm.sum(e_exact))
    e_piml_total = float(bm.sum(e_piml))
    e_rigid_total = float(bm.sum(e_rigid))

    return {
        "n_rigid_modes": n_rigid,
        "exact_rigid_eigenvalue_max": float(bm.max(lam_rigid)),
        "exact_soft_eigenvalue_min": float(bm.min(lam_soft)),
        "exact_rigid_residual_max": float(bm.max(pol_exact)),
        "rigid_pollution_max": float(bm.max(pol_piml)),
        "rigid_pollution_ratio_max": float(bm.max(pol_ratio)),
        "rigid_pollution_ratio_mean": float(bm.mean(pol_ratio)),
        "rigid_displacement_fraction_mean": float(bm.mean(rigid_fraction)),
        "energy_exact": e_exact_total,
        "energy_piml": e_piml_total,
        "energy_stiffening_factor": e_piml_total / e_exact_total,
        "energy_rigid_pollution_share": e_rigid_total / e_exact_total,
    }


### 验收与落盘 ###

def validate_and_write_result(
    result: Dict[str, Any], output_dir: str | None, strict: bool
) -> None:
    """报告对比结果, 并可选地按严格模式验收与落盘.

    参数:
        result: 本次运行的全部统计指标.
        output_dir: 证据输出目录; 为 ``None`` 时只验收不落盘.
        strict: 为 ``True`` 时要求全部子结构都真正使用代理预测.

    异常:
        AssertionError: ``strict`` 为 ``True`` 且在役子结构或留出样本上存在回退时抛出.

    说明:
        代理精度不设阈值断言: 拟合误差随训练配置连续变化, 任何硬编阈值都只是伪门禁.
        唯一可判定的是代理是否真正生效, 这由回退计数表达.

        在役的 24 个子结构只覆盖一组特定密度, 留出集则覆盖整个训练分布. 两者都纳入
        严格模式: 仅前者通过而后者失败, 说明代理在分布内存在成片的退化预测, 只是恰好
        没落在本次评估的密度上.
    """
    if strict and result["n_fallback"] > 0:
        raise AssertionError(
            f"{result['n_fallback']}/{result['n_substructures']} 个子结构回退到精确缩聚, "
            f"代理未真正生效"
        )

    if strict and result.get("holdout_n_fallback", 0) > 0:
        raise AssertionError(
            f"{result['holdout_n_fallback']}/{result['holdout_samples']} 组留出样本"
            f"回退到精确缩聚, 代理在训练分布内存在退化预测"
        )

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "piml_exact_comparison.json"
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[证据] 结果已写入: {target}")


### 云图 ###

def plot_comparison(
    assembler: GlobalAssembler,
    u_full_exact: Any,
    u_full_piml: Any,
    K_s_exact: Any,
    K_s_piml: Any,
    domain_size: Sequence[float],
    fig_path: Path,
) -> None:
    """绘制两条路径的位移场与首个子结构的缩聚刚度热图.

    参数:
        assembler: 全局装配器, 提供节点值到网格的重排.
        u_full_exact: 精确缩聚路径的全场位移.
        u_full_piml: 代理缩聚路径的全场位移.
        K_s_exact: 精确缩聚刚度批量, 形状 ``(B, n_b, n_b)``.
        K_s_piml: 代理缩聚刚度批量, 形状 ``(B, n_b, n_b)``.
        domain_size: 求解域尺寸 ``(Lx, Ly)``.
        fig_path: 图像落盘路径.
    """
    Lx, Ly = domain_size[0], domain_size[1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # to_node_grid 依据节点坐标重排, 不假定网格生成器的节点编号次序.
    for ax, (title, u_full) in zip(
        axes[0],
        (
            ("Exact Schur Condensation: U_y", u_full_exact),
            ("PIML Surrogate Condensation: U_y", u_full_piml),
        ),
    ):
        field = bm.to_numpy(assembler.to_node_grid(u_full[1::2])).T
        im = ax.imshow(
            field, origin="lower", cmap="viridis", extent=[0.0, Lx, 0.0, Ly]
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    for ax, (title, K_s) in zip(
        axes[1],
        (
            ("Exact K_s (substructure 0)", K_s_exact),
            ("PIML K_s (substructure 0)", K_s_piml),
        ),
    ):
        im = ax.imshow(bm.to_numpy(K_s[0]), cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


### 主流程 ###

def run_comparison(
    n_train: int,
    n_epochs: int,
    learning_rate: float,
    n_val: int,
    seed: int,
    output_dir: str | None,
    strict: bool,
    backend: Literal["numpy", "pytorch"],
) -> Dict[str, Any]:
    """运行 PIML 代理缩聚与精确缩聚的端到端对比.

    参数:
        n_train: 随机密度训练样本数.
        n_epochs: 训练轮数.
        learning_rate: Adam 学习率.
        n_val: 训练同分布留出样本数, 用于区分欠拟合与分布错配.
        seed: 随机数种子, 覆盖训练采样, 留出采样与网络初始化.
        output_dir: 证据输出目录; 为 ``None`` 时不落盘.
        strict: 为 ``True`` 时要求代理全程生效.
        backend: 本次运行使用的 ``bm`` 后端. 后端选择仅位于示例入口, 不传入核心库.

    返回:
        result: 本次运行的全部统计指标, 与落盘 JSON 同构.
    """
    bm.set_backend(backend)
    set_random_seed(seed)

    problem = FullMBBBeam2d(domain=(0.0, 12.0, 0.0, 2.0), P=-1.0, E=1.0, nu=0.3)
    domain_size = (problem.domain[1], problem.domain[3])
    n_sub = (12, 2)
    n_fine = (5, 5)

    assembler = GlobalAssembler(
        domain_size, n_sub, n_fine, E_base=problem.E, nu=problem.nu
    )
    prototype, sub_meshes, positions = build_substructures(assembler)
    n_substructures = len(sub_meshes)

    print("=" * 86)
    print("PIML 代理缩聚 vs 精确 Schur 补缩聚 (FullMBBBeam2d, Huang 2023 第 4.1 节)")
    print("=" * 86)
    print(f"求解域         : {domain_size[0]} x {domain_size[1]}")
    print(f"子结构划分     : {n_sub[0]} x {n_sub[1]} (共 {n_substructures} 个)")
    print(f"子结构细网格   : {n_fine[0]} x {n_fine[1]} Q4 单元")
    print(f"接口自由度     : 单个子结构 {prototype.n_b}, 内部 {prototype.n_i}")
    print(f"随机数种子     : {seed}")
    print("-" * 86)

    density = make_density_fields(sub_meshes, domain_size)

    # 分析器在此只作为外载与约束的来源, 既不装配全局刚度也不求解全尺度问题: 精确缩聚
    # 与全装配的等价性已由 examples/substructure_elasticity/compare_lagrange.py 建立.
    # 外载取 assemble_external_load(), 即施加 Dirichlet 条件之前的体力与 Neumann
    # 等效节点力之和; force_vector 属性在 apply_bc 之前为 None, 不能在此使用.
    analyzer = LagrangeFEMAnalyzer(
        disp_mesh=assembler.full_mesh,
        pde=problem,
        material=assembler.material,
        space_degree=1,
        operator_level="fa",
        topopt_algorithm="density_based",
        solve_method="scipy",
        interpolation_scheme=MaterialInterpolationScheme(
            density_location="element",
            interpolation_method="simp",
            options={"penalty_factor": 3.0, "stress_penalty_factor": 1.0},
        ),
        enable_logging=False,
    )
    global_load = bm.asarray(analyzer.assemble_external_load(), dtype=bm.float64)
    _, fixed_mask = analyzer.tensor_space.boundary_interpolate(
        gd=problem.dirichlet_bc,
        threshold=cast(Any, problem.is_dirichlet_boundary()),
        method="interp",
    )
    fixed_global_dofs = bm.nonzero(fixed_mask)[0]

    # ------------------------------------------------------------------
    # 路径 A: 精确批量 Schur 补缩聚
    # ------------------------------------------------------------------
    print("[路径 A] 精确 Schur 补批量缩聚...")
    K_local_batch = prototype.assemble_local_stiffness_batch(density)
    t0 = time.time()
    exact_condensor = FEAStaticCondensation(prototype.i_dofs, prototype.b_dofs)
    K_s_exact, _ = exact_condensor.condense(K_local_batch)
    t_exact = time.time() - t0

    u_b_exact, u_full_exact, n_free = solve_with_condensors(
        assembler, sub_meshes, exact_condensor, global_load, fixed_global_dofs
    )

    # ------------------------------------------------------------------
    # 代理训练
    # ------------------------------------------------------------------
    # 先确认刚体基正确且训练目标与推理重构是同一个参数化, 否则后续精度数字不可解释.
    parity = verify_parameterization_parity(prototype)
    print(
        f"[自检]   刚体基残差 {parity['rigid_basis_residual']:.4e}, "
        f"完美代理下 K_s 相对误差上限 {parity['parameterization_error_ceiling']:.4e}"
    )

    print(f"[训练]   {n_train} 组随机密度, {n_epochs} 轮...")
    net, final_loss = train_surrogate(prototype, n_train, n_epochs, learning_rate)
    print(f"         最终训练 MSE: {final_loss:.6e}")

    # ------------------------------------------------------------------
    # 路径 B: PIML 代理缩聚
    # ------------------------------------------------------------------
    print("[路径 B] PIML 代理逐子结构缩聚...")
    t0 = time.time()
    piml_condensors: List[PIMLStaticCondensation] = []
    for idx, sub_mesh in enumerate(sub_meshes):
        condensor = PIMLStaticCondensation(
            sub_mesh.i_dofs, sub_mesh.b_dofs, model=net, is_cholesky=True,
            range_basis=sub_mesh.deformation_basis,
        )
        condensor.condense(K_local_batch[idx], density[idx])
        piml_condensors.append(condensor)
    t_piml = time.time() - t0

    n_fallback = sum(1 for c in piml_condensors if c.used_fallback)
    K_s_piml = bm.stack([c.K_s for c in piml_condensors], axis=0)

    u_b_piml, u_full_piml, _ = solve_with_condensors(
        assembler, sub_meshes, piml_condensors, global_load, fixed_global_dofs
    )

    # ------------------------------------------------------------------
    # 算子层与解层指标
    # ------------------------------------------------------------------
    err_ks_each = bm.linalg.norm(
        K_s_piml - K_s_exact, axis=(-2, -1)
    ) / bm.linalg.norm(K_s_exact, axis=(-2, -1))
    err_ks_max = float(bm.max(err_ks_each))
    err_ks_mean = float(bm.mean(err_ks_each))

    err_u_b = float(
        bm.linalg.norm(u_b_piml - u_b_exact) / bm.linalg.norm(u_b_exact)
    )
    err_u_full = float(
        bm.linalg.norm(u_full_piml - u_full_exact) / bm.linalg.norm(u_full_exact)
    )

    # 柔度 C = f^T u, 两条路径共用同一份外载向量.
    c_exact = float(bm.sum(global_load * u_full_exact))
    c_piml = float(bm.sum(global_load * u_full_piml))
    err_c = abs(c_piml - c_exact) / abs(c_exact)

    # ------------------------------------------------------------------
    # 误差归因诊断
    # ------------------------------------------------------------------
    print(f"[诊断]   {n_val} 组训练同分布留出样本...")
    holdout = evaluate_holdout(prototype, net, n_val)

    # 精确解在各子结构接口自由度上的取值, 列序与该子结构的 b_dofs 一致.
    b_global = bm.stack(
        [
            assembler.get_substructure_global_dofs(*pos, sm)[sm.b_dofs]
            for pos, sm in zip(positions, sub_meshes)
        ],
        axis=0,
    )
    dim = assembler.dim
    rigid = diagnose_rigid_mode_pollution(
        K_s_exact, K_s_piml, u_full_exact[b_global], dim * (dim + 1) // 2
    )

    result: Dict[str, Any] = {
        "problem": type(problem).__name__,
        "domain": list(problem.domain),
        "n_sub": list(n_sub),
        "n_fine": list(n_fine),
        "n_substructures": n_substructures,
        "n_interface_dofs_local": int(prototype.n_b),
        "n_interior_dofs_local": int(prototype.n_i),
        "n_free_interface_dofs": n_free,
        "n_train_samples": n_train,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "final_train_mse": final_loss,
        "n_reduced_dofs": int(prototype.deformation_basis.shape[1]),
        "density_range": list(DENSITY_RANGE),
        "n_fallback": n_fallback,
        "ks_relative_error_max": err_ks_max,
        "ks_relative_error_mean": err_ks_mean,
        "interface_displacement_relative_error": err_u_b,
        "displacement_relative_error": err_u_full,
        "compliance_exact": c_exact,
        "compliance_piml": c_piml,
        "compliance_relative_error": err_c,
        "condensation_time_exact": t_exact,
        "condensation_time_piml": t_piml,
        "backend": backend,
        "strict": strict,
    }
    result.update(parity)
    result.update(holdout)
    result.update(rigid)

    # ------------------------------------------------------------------
    # 结果表格
    # ------------------------------------------------------------------
    print("\n" + "=" * 86)
    print(format_table_row("指标", "精确 Schur 补", "PIML 代理"))
    print("-" * 86)
    print(format_table_row("缩聚接口求解自由度", str(n_free), str(n_free)))
    print(format_table_row(
        "局部缩聚耗时 (s)", f"{t_exact:.4f} (批量)", f"{t_piml:.4f} (逐个)"
    ))
    print(format_table_row("回退到精确缩聚的子结构数", "—", f"{n_fallback}/{n_substructures}"))
    print("-" * 86)
    print(format_table_row("算子层 | K_s 相对误差 (max)", "0", f"{err_ks_max:.4e}"))
    print(format_table_row("算子层 | K_s 相对误差 (mean)", "0", f"{err_ks_mean:.4e}"))
    print(format_table_row("解层   | 接口位移相对误差", "0", f"{err_u_b:.4e}"))
    print(format_table_row("解层   | 全场位移相对误差", "0", f"{err_u_full:.4e}"))
    print(format_table_row("解层   | 柔度", f"{c_exact:.8f}", f"{c_piml:.8f}"))
    print(format_table_row("解层   | 柔度相对误差", "0", f"{err_c:.4e}"))
    print("=" * 86)

    if n_fallback > 0:
        print(
            f"[警告] {n_fallback}/{n_substructures} 个子结构回退到精确缩聚, "
            f"上表中这些子结构的 PIML 列即精确解."
        )
    if err_ks_max > 0.0:
        print(f"误差放大倍率 (全场位移 / K_s max): {err_u_full / err_ks_max:.3f}")

    # ------------------------------------------------------------------
    # 诊断报告
    # ------------------------------------------------------------------
    err_holdout_mean = holdout["holdout_ks_relative_error_mean"]
    print("\n" + "=" * 86)
    print("误差归因诊断")
    print("-" * 86)
    print(format_diag_row(
        "[参数化] 刚体基残差 ||K_s R||/||K_s|| (应 ≈0)",
        f"{parity['rigid_basis_residual']:.4e}",
    ))
    print(format_diag_row(
        "[参数化] 完美代理下的 K_s 相对误差上限",
        f"{parity['parameterization_error_ceiling']:.4e}  "
        f"(观测误差高于它多少倍即网络拟合的责任)",
    ))
    print("[分布错配] 训练分布内 vs 光滑评估场")
    print(format_diag_row(
        f"  留出集 K_s 相对误差 (max / mean), {n_val} 组",
        f"{holdout['holdout_ks_relative_error_max']:.4e} / {err_holdout_mean:.4e}",
    ))
    print(format_diag_row(
        "  光滑评估场 K_s 相对误差 (max / mean)",
        f"{err_ks_max:.4e} / {err_ks_mean:.4e}",
    ))
    if err_holdout_mean > 0.0:
        print(format_diag_row(
            "  光滑场 / 留出集 (mean 之比)",
            f"{err_ks_mean / err_holdout_mean:.3f}  "
            f"(≈1 则瓶颈是欠拟合, ≫1 则是分布错配)",
        ))
    if holdout["holdout_n_fallback"] > 0:
        print(format_diag_row(
            "  留出集回退计数",
            f"{holdout['holdout_n_fallback']}/{n_val}",
        ))

    print("[零空间污染] 参数化已限制在刚体模态的正交补上, 伪刚度应降到机器精度")
    print(format_diag_row(
        f"  精确 K_s 刚体特征值 (max, 应 ≈0), {rigid['n_rigid_modes']} 个",
        f"{rigid['exact_rigid_eigenvalue_max']:.4e}",
    ))
    print(format_diag_row(
        "  精确 K_s 最小非零特征值 (min)",
        f"{rigid['exact_soft_eigenvalue_min']:.4e}",
    ))
    print(format_diag_row(
        "  代理在刚体子空间上的伪刚度 (max)",
        f"{rigid['rigid_pollution_max']:.4e}",
    ))
    print(format_diag_row(
        "  伪刚度 / 最小非零特征值 (max / mean)",
        f"{rigid['rigid_pollution_ratio_max']:.4e} / "
        f"{rigid['rigid_pollution_ratio_mean']:.4e}",
    ))

    print("[能量归因] 在精确解 u_b 上求值")
    print(format_diag_row(
        "  应变能 Σ u_b^T K_s u_b  精确 / 代理",
        f"{rigid['energy_exact']:.6e} / {rigid['energy_piml']:.6e}",
    ))
    print(format_diag_row(
        "  代理刚化倍率",
        f"{rigid['energy_stiffening_factor']:.4f}",
    ))
    print(format_diag_row(
        "  其中刚体子空间伪刚度贡献占精确应变能",
        f"{rigid['energy_rigid_pollution_share']:.4f}",
    ))
    print(format_diag_row(
        "  精确解位移落在刚体子空间的比例 (mean)",
        f"{rigid['rigid_displacement_fraction_mean']:.4f}",
    ))
    print("=" * 86)

    if output_dir is not None:
        fig_path = Path(output_dir)
        fig_path.mkdir(parents=True, exist_ok=True)
        target = fig_path / "piml_exact_comparison.png"
        plot_comparison(
            assembler, u_full_exact, u_full_piml, K_s_exact, K_s_piml,
            domain_size, target,
        )
        print(f"[产物] 对比云图已保存: {target}")

    validate_and_write_result(result, output_dir, strict)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIML 代理缩聚与精确 Schur 补缩聚对比"
    )
    parser.add_argument(
        "--train-samples", type=int, default=300, help="随机密度训练样本数"
    )
    parser.add_argument("--epochs", type=int, default=400, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam 学习率")
    parser.add_argument(
        "--val-samples",
        type=int,
        default=100,
        help="训练同分布留出样本数, 用于区分欠拟合与分布错配",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="随机数种子, 覆盖训练采样, 留出采样与网络初始化",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="要求代理全程生效, 出现回退即以异常失败",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("outputs")),
        help="写入 JSON 证据与对比云图的目录",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "pytorch"),
        default="numpy",
        help="bm 后端, 默认 numpy",
    )
    args = parser.parse_args()

    run_comparison(
        n_train=args.train_samples,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        n_val=args.val_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        strict=args.strict,
        backend=cast(Literal["numpy", "pytorch"], args.backend),
    )


if __name__ == "__main__":
    main()
