"""四范式对比的统一度量.

本模块是四条求解路径共用的唯一度量实现. 现有三个成对比较脚本各自报告不同的量
(``examples/substructure_elasticity/compare_lagrange.py`` 报收敛阶,
``examples/piml_substructure_elasticity/verify_stiffness_route.py`` 报 ``K_s`` 的 Frobenius
误差, 同目录 ``compare_piml_pinn.py`` 报全场位移 ``L2`` 误差), 无法横向对齐; 本模块
把它们收敛为一套口径, 各路径只负责产出位移场与算子, 不自带误差定义.

刚体模态基不在此构造: ``SubstructurePrototype.rigid_basis`` 已提供, 相关度量按参数接收.
"""

from __future__ import annotations

from typing import Any

from fealpy.backend import backend_manager as bm


def relative_l2(u: Any, u_ref: Any) -> float:
    """计算位移场的相对 ``L2`` 误差.

    参数:
        u: 待评价位移向量, 形状 ``(n_dof,)``.
        u_ref: 参考位移向量, 形状 ``(n_dof,)``.

    返回:
        ``||u - u_ref||_2 / ||u_ref||_2``.

    异常:
        ValueError: 当两向量形状不一致或参考解为零向量时抛出.
    """
    u, u_ref = _as_vector(u), _as_vector(u_ref)
    if u.shape != u_ref.shape:
        raise ValueError(f"位移向量形状不一致: {u.shape} 与 {u_ref.shape}.")
    denominator = float(bm.sqrt(bm.sum(u_ref * u_ref)))
    if denominator == 0.0:
        raise ValueError("参考位移为零向量, 相对误差无定义.")
    return float(bm.sqrt(bm.sum((u - u_ref) ** 2))) / denominator


def subset_relative_l2(u: Any, u_ref: Any, dofs: Any) -> float:
    """在指定自由度子集上计算相对 ``L2`` 误差.

    用于把误差分层到接口自由度与内部恢复自由度: 缩聚类路径的接口误差来自代理算子,
    内部误差还叠加了恢复矩阵 ``N`` 的误差, 两者混在一起会掩盖误差来源.

    参数:
        u: 待评价位移向量, 形状 ``(n_dof,)``.
        u_ref: 参考位移向量, 形状 ``(n_dof,)``.
        dofs: 自由度索引数组, 形状 ``(n_selected,)``.

    返回:
        子集上的相对 ``L2`` 误差.
    """
    index = bm.asarray(dofs, dtype=bm.int64)
    return relative_l2(_as_vector(u)[index], _as_vector(u_ref)[index])


def energy_norm_relative_error(u: Any, u_ref: Any, K: Any) -> float:
    """计算能量范数下的相对误差.

    Schur 补缩聚在数学上是能量二次型上的 Ritz 投影, 因此能量范数是缩聚类路径最自然的
    误差度量; 对 PINN 与全装配路径同样有定义, 故取为四条路径的公共主指标.

    参数:
        u: 待评价位移向量, 形状 ``(n_dof,)``.
        u_ref: 参考位移向量, 形状 ``(n_dof,)``.
        K: 参考路径的全局刚度矩阵, 支持稠密数组或实现了 ``@`` 的稀疏矩阵,
            形状 ``(n_dof, n_dof)``.

    返回:
        ``sqrt(e^T K e) / sqrt(u_ref^T K u_ref)``, 其中 ``e = u - u_ref``.

    异常:
        ValueError: 当参考解能量为零, 或误差能量为负 (刚度矩阵非半正定) 时抛出.
    """
    u, u_ref = _as_vector(u), _as_vector(u_ref)
    error = u - u_ref
    numerator = float(bm.sum(error * (K @ error)))
    denominator = float(bm.sum(u_ref * (K @ u_ref)))
    if denominator <= 0.0:
        raise ValueError("参考解应变能非正, 能量范数相对误差无定义.")
    if numerator < 0.0:
        raise ValueError(f"误差应变能为负 ({numerator:.3e}), 刚度矩阵非半正定.")
    return (numerator / denominator) ** 0.5


def compliance(f: Any, u: Any) -> float:
    """计算结构柔顺度 ``C = f^T u``.

    参数:
        f: 全局外载向量, 形状 ``(n_dof,)``.
        u: 全场位移向量, 形状 ``(n_dof,)``.

    返回:
        柔顺度标量.
    """
    return float(bm.sum(_as_vector(f) * _as_vector(u)))


def relative_error(value: float, reference: float) -> float:
    """计算标量的相对误差.

    参数:
        value: 待评价标量.
        reference: 参考标量.

    返回:
        ``|value - reference| / |reference|``.

    异常:
        ValueError: 当参考标量为零时抛出.
    """
    if reference == 0.0:
        raise ValueError("参考标量为零, 相对误差无定义.")
    return abs(value - reference) / abs(reference)


def min_eigenvalue(K_s: Any) -> float:
    """返回缩聚刚度矩阵的最小特征值.

    仅对 PIML 路径有意义: 精确 Schur 补由构造保证半正定, 代理预测则可能穿零.
    该值是结构保持退化曲线的纵坐标, 也是在线回退门禁的判据.

    参数:
        K_s: 对称缩聚刚度矩阵, 形状 ``(n_b, n_b)`` 或批量 ``(B, n_b, n_b)``.

    返回:
        全部子结构中的最小特征值.
    """
    return float(bm.min(bm.linalg.eigvalsh(bm.asarray(K_s))))


def rigid_mode_residual(K_s: Any, rigid_basis: Any) -> float:
    """计算缩聚刚度矩阵对刚体模态的相对残差.

    精确 Schur 补严格保持刚体零空间, 即 ``K_s R = 0``; 代理刚度不满足该性质时会在
    全局接口系统中引入伪刚度, 该残差量化这一污染.

    参数:
        K_s: 对称缩聚刚度矩阵, 形状 ``(n_b, n_b)``.
        rigid_basis: 刚体模态基, 取自 ``SubstructurePrototype.rigid_basis``,
            形状 ``(n_b, n_rigid)``.

    返回:
        ``||K_s R||_F / ||K_s||_F``.

    异常:
        ValueError: 当 ``K_s`` 为零矩阵时抛出.
    """
    K_s = bm.asarray(K_s)
    R = bm.asarray(rigid_basis)
    scale = float(bm.sqrt(bm.sum(K_s * K_s)))
    if scale == 0.0:
        raise ValueError("缩聚刚度为零矩阵, 刚体模态相对残差无定义.")
    residual = K_s @ R
    return float(bm.sqrt(bm.sum(residual * residual))) / scale


def breakeven_count(
    offline_seconds: float,
    online_seconds: float,
    reference_seconds: float,
) -> float:
    """计算相对参考路径的摊销盈亏点 ``N*``.

    单次求解计时对含离线训练的路径不公平, 对训练即求解的 PINN 同样不公平. 统一口径
    取求解 ``N`` 个宏观边值问题的总时间: 参考路径为 ``N * reference_seconds``,
    代理路径为 ``offline_seconds + N * online_seconds``, ``N*`` 是两者相等的问题数.

    参数:
        offline_seconds: 一次性离线训练耗时; 无离线阶段的路径取 ``0.0``.
        online_seconds: 单个边值问题的在线求解耗时; PINN 每换一次定解问题都要重训,
            故其在线耗时即单次训练耗时.
        reference_seconds: 参考路径求解单个边值问题的耗时.

    返回:
        盈亏点问题数; 在线耗时不低于参考路径时返回 ``float('inf')``, 表示永不回本.
    """
    margin = reference_seconds - online_seconds
    if margin <= 0.0:
        return float("inf")
    return offline_seconds / margin


def _as_vector(values: Any) -> Any:
    """把输入统一为一维 ``float64`` 向量.

    参数:
        values: 位移或载荷数组, 形状 ``(n_dof,)`` 或可展平为该形状.

    返回:
        一维 ``float64`` 数组.
    """
    array = bm.asarray(values, dtype=bm.float64)
    return bm.reshape(array, (-1,))
