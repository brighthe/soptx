"""同构子结构示例与验证辅助工具 (Example Utilities).

说明:
    本模块存放供 `examples/` 与验证脚本复用的算例辅助搭建与训练胶水代码.
    通用的有限元网格铺设请使用 `soptx.fem.substructure.mesh.build_substructures`,
    神经网络模型请使用 `soptx.ml.substructure_nets`.
"""

from typing import Any, Callable, List, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim

from fealpy.backend import backend_manager as bm

from .mesh import SubstructureMesh, SubstructurePrototype, build_substructures
from .condensation import FEAStaticCondensation
from .piml_surrogate import PIMLSurrogateNet
from .assembler import GlobalAssembler


def set_random_seed(seed: int) -> None:
    """统一固定 ``bm`` 后端与 PyTorch 的随机数种子.

    参数:
        seed: 随机数种子.

    说明:
        必须在 ``bm.set_backend`` 之后调用: ``bm.random`` 的实现由当前后端决定.
        pytorch 后端下 ``bm.random`` 映射为 ``torch.random``, 其 ``seed()`` 不接受
        参数 (带参数的入口是 ``torch.manual_seed``), 因此只在 numpy 后端调用
        ``bm.random.seed``; pytorch 后端的全部随机源统一由 ``torch.manual_seed``
        固定.
    """
    if bm.backend_name == "numpy":
        bm.random.seed(seed)
    torch.manual_seed(seed)


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
    if assembler.dim == 2:
        for sx in range(assembler.n_sub[0]):
            for sy in range(assembler.n_sub[1]):
                spans = (
                    (sx * sub_size[0], (sx + 1) * sub_size[0]),
                    (sy * sub_size[1], (sy + 1) * sub_size[1]),
                )
                sub_meshes.append(
                    SubstructureMesh(
                        sub_id, *spans, *assembler.n_fine,
                        E_base=assembler.E_base, nu=assembler.nu, prototype=prototype,
                    )
                )
                positions.append((sx, sy))
                sub_id += 1
    elif assembler.dim == 3:
        for sx in range(assembler.n_sub[0]):
            for sy in range(assembler.n_sub[1]):
                for sz in range(assembler.n_sub[2]):
                    spans = (
                        (sx * sub_size[0], (sx + 1) * sub_size[0]),
                        (sy * sub_size[1], (sy + 1) * sub_size[1]),
                        (sz * sub_size[2], (sz + 1) * sub_size[2]),
                    )
                    sub_meshes.append(
                        SubstructureMesh(
                            sub_id, *spans, *assembler.n_fine,
                            E_base=assembler.E_base, nu=assembler.nu, prototype=prototype,
                        )
                    )
                    positions.append((sx, sy, sz))
                    sub_id += 1
    return prototype, sub_meshes, positions


def make_density_fields(
    sub_meshes: Sequence[SubstructureMesh],
    domain_size: Sequence[float],
    density_range: Tuple[float, float],
) -> Any:
    """生成逐细单元变化的批量局部密度场.

    参数:
        sub_meshes: 子结构列表, 全部同构.
        domain_size: 求解域在各方向的尺寸 ``(Lx, Ly)`` 或 ``(Lx, Ly, Lz)``.
        density_range: 密度取值区间 ``(lo, hi)``; 与代理配合使用时应传训练分布的
            区间, 保证评估工作在内插而非外插区间.

    返回:
        density: 形状 ``(B, nx, ny)`` 或 ``(B, nx, ny, nz)`` 的批量密度场.
    """
    n_fine = tuple(sub_meshes[0].n_fine)
    dim = len(domain_size)

    lows = bm.asarray(
        [[sm.box_span[d][0] for d in range(dim)] for sm in sub_meshes],
        dtype=bm.float64,
    )
    spans = bm.asarray(
        [[sm.box_span[d][1] - sm.box_span[d][0] for d in range(dim)] for sm in sub_meshes],
        dtype=bm.float64,
    )

    lo, hi = density_range
    mid = 0.5 * (lo + hi)
    amp = 0.45 * (hi - lo)

    if dim == 2:
        unit_x = (bm.arange(n_fine[0], dtype=bm.float64) + 0.5) / n_fine[0]
        unit_y = (bm.arange(n_fine[1], dtype=bm.float64) + 0.5) / n_fine[1]
        x = (lows[:, 0:1] + unit_x[None, :] * spans[:, 0:1])[:, :, None]
        y = (lows[:, 1:2] + unit_y[None, :] * spans[:, 1:2])[:, None, :]
        modulation = bm.sin(bm.pi * x / domain_size[0]) * bm.cos(bm.pi * y / domain_size[1])
        return mid + amp * modulation
    elif dim == 3:
        unit_x = (bm.arange(n_fine[0], dtype=bm.float64) + 0.5) / n_fine[0]
        unit_y = (bm.arange(n_fine[1], dtype=bm.float64) + 0.5) / n_fine[1]
        unit_z = (bm.arange(n_fine[2], dtype=bm.float64) + 0.5) / n_fine[2]
        x = (lows[:, 0:1] + unit_x[None, :] * spans[:, 0:1])[:, :, None, None]
        y = (lows[:, 1:2] + unit_y[None, :] * spans[:, 1:2])[:, None, :, None]
        z = (lows[:, 2:3] + unit_z[None, :] * spans[:, 2:3])[:, None, None, :]
        modulation = (
            bm.sin(bm.pi * x / domain_size[0])
            * bm.cos(bm.pi * y / domain_size[1])
            * bm.cos(bm.pi * z / domain_size[2])
        )
        return mid + amp * modulation


def sample_random_density(
    prototype: SubstructurePrototype,
    n_sample: int,
    density_range: Tuple[float, float],
) -> Any:
    """按训练分布采样一批随机局部密度.

    参数:
        prototype: 共享参考子结构, 提供细网格规模.
        n_sample: 采样组数.
        density_range: 密度取值区间 ``(lo, hi)``.

    返回:
        rho: 形状 ``(n_sample, nx, ny)`` 的密度, 各分量在 ``density_range`` 上独立
            均匀采样.

    说明:
        训练集与留出集共用本函数, 以此保证两者严格同分布——留出误差与光滑评估场误差
        的对比只有在这一前提下才能归因于分布错配.
    """
    # 后端在运行时接受多个尺寸参数, 但静态类型声明仅暴露单个参数.
    random_rand = cast(Callable[..., Any], bm.random.rand)
    lo, hi = density_range
    return lo + (hi - lo) * bm.asarray(
        random_rand(n_sample, *tuple(prototype.n_fine)), dtype=bm.float64
    )


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
    density_range: Tuple[float, float],
) -> Tuple[PIMLSurrogateNet, float]:
    """在随机密度样本上训练 Cholesky 因子代理网络.

    参数:
        prototype: 共享参考子结构, 提供细网格规模与自由度划分.
        n_train: 随机密度训练样本数.
        n_epochs: 全批量梯度下降的迭代轮数.
        learning_rate: Adam 学习率.
        density_range: 训练分布的密度取值区间 ``(lo, hi)``.

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

    rand_rho = sample_random_density(prototype, n_train, density_range)

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
