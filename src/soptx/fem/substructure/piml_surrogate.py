"""基于神经网络代理的子结构静力缩聚.

代理只替换缩聚刚度 ``K_s``; 内部位移恢复矩阵 ``N`` 恒由局部刚度精确导出. 预测异常
或未通过门禁时自动回退精确 Schur 补, 并置 ``used_fallback``.

Cholesky 参数化在**变形子空间**上进行: 自由漂浮子结构的 ``K_s`` 以刚体模态为精确
零空间, 把预测限制在其正交补上, 使秩亏结构成为构造性质. 数学背景与实测证据见
``docs/fem/substructure-condensation-implementation.md``.

所有数组与代数运算基于 FEALPy 后端管理器 (bm).
"""

from typing import Tuple, Any, Optional
import torch
import torch.nn as nn
from fealpy.backend import backend_manager as bm

from soptx.ml import MLP

from .condensation import StaticCondensationBase, FEAStaticCondensation


class SurrogateContractError(RuntimeError):
    """代理网络与参数化的接口契约不符.

    与数值退化不同, 该错误表示配置本身有误 (例如网络输出维与独立条目数不匹配),
    静默回退只会把它掩盖成一次精度下降, 因此不参与回退, 直接向上抛出.
    """


class PIMLSurrogateNet(MLP):
    """用于预测 Cholesky 下三角独立条目的 PyTorch MLP 代理网络."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        """初始化 PIML 子结构刚度代理网络.

        参数:
            input_dim: 展平后的子结构单元密度特征数.
            output_dim: Cholesky 下三角因子独立条目的数量.
            hidden_dim: 两个 SiLU 隐藏层的共同宽度.
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            activation=nn.SiLU,
        )


class PIMLStaticCondensation(StaticCondensationBase):
    """基于 PIML 神经网络代理的子结构静力缩聚.

    ``is_cholesky=True`` 时按变形子空间上的 Cholesky 因子参数化:
    网络输出 ``m(m+1)/2`` 个下三角条目构成 ``L``, 由

        ``K_s = R_perp @ L @ L^T @ R_perp^T``

    重构缩聚刚度, 其中 ``R_perp`` 是接口自由度上刚体模态的标准正交补, 形状
    ``(n_b, m)``, ``m = n_b - n_rigid``. 该参数化同时给出三条构造性质:

    1. ``K_s`` 在刚体子空间上恒为零, 拟合误差无法泄漏进刚体模态. 装配后各子结构
       的接口位移几乎全部是刚体运动, 该子空间上的伪刚度会被平方量级地放大, 因此
       这一条决定解的精度.
    2. ``K_s`` 在变形子空间上正定, 与精确 ``K_s`` 的秩亏结构一致.
    3. 训练目标 ``cholesky(R_perp^T K_s R_perp)`` 无需正则化: 限制后的算子严格
       正定且条件数适中, 由此消除了历史实现中 ``1e-6`` 正则带来的系统性正偏置.

    ``is_cholesky=False`` 时网络输出被解释为对称矩阵的上三角条目, 无任何结构保证,
    正定性完全由训练结果与门禁承担.

    门禁按参数化分工: Cholesky 路径检查预测是否有限, 以及 ``L`` 的对角线是否近乎
    退化 (最小与最大绝对值之比低于 ``rcond_min``); 对称路径检查最小特征值是否为正.
    Cholesky 路径不对 ``K_s`` 求特征值——它按构造秩亏, 最小特征值恒为零, 用正定性
    判据会导致恒定回退.
    """

    def __init__(
        self,
        i_dofs: Any,
        b_dofs: Any,
        model: Optional[nn.Module] = None,
        is_cholesky: bool = True,
        range_basis: Optional[Any] = None,
        rcond_min: float = 1.0e-8,
    ) -> None:
        """初始化 PIML 静力缩聚器.

        参数:
            i_dofs: 子结构内部自由度的局部编号.
            b_dofs: 子结构接口自由度的局部编号.
            model: 从局部密度预测缩聚刚度参数的代理网络. 为 ``None`` 时总是回退
                精确缩聚.
            is_cholesky: 是否将网络输出解释为变形子空间上 Cholesky 因子的下三角
                条目.
            range_basis: 变形子空间的标准正交基 ``R_perp``, 形状 ``(n_b, m)``, 通常
                取 ``SubstructurePrototype.deformation_basis``. ``is_cholesky`` 为
                ``True`` 时必须给出.
            rcond_min: ``L`` 对角线的最小与最大绝对值之比的下限, 低于它判定为数值
                退化并回退.

        异常:
            ValueError: ``is_cholesky`` 为 ``True`` 而未给出 ``range_basis``, 或
                ``range_basis`` 的行数与 ``n_b`` 不符时抛出.
        """
        super(PIMLStaticCondensation, self).__init__(i_dofs, b_dofs)
        self.model = model
        self.is_cholesky = is_cholesky
        self.rcond_min = float(rcond_min)

        # 精确缩聚器是模型缺失, 预测失败或门禁失败时的回退路径.
        self.fallback_solver = FEAStaticCondensation(i_dofs, b_dofs)
        self.used_fallback = False

        # 掩码按行优先枚举独立条目, 与训练侧 ``L[tril_mask]`` 的取值顺序一致.
        if self.is_cholesky:
            if range_basis is None:
                raise ValueError(
                    "Cholesky 参数化需要变形子空间的正交基 range_basis; "
                    "通常取 SubstructurePrototype.deformation_basis."
                )
            basis = bm.asarray(range_basis, dtype=bm.float64)
            if basis.shape[0] != self.n_b:
                raise ValueError(
                    f"range_basis 的行数应为接口自由度数 {self.n_b}; "
                    f"当前形状为 {tuple(basis.shape)}."
                )
            self.range_basis: Any = basis
            self.n_reduced: int = int(basis.shape[1])
            self.entry_mask: Any = bm.tril(
                bm.ones((self.n_reduced, self.n_reduced), dtype=bm.bool)
            )
        else:
            self.range_basis = None
            self.n_reduced = self.n_b
            self.entry_mask = bm.triu(
                bm.ones((self.n_b, self.n_b), dtype=bm.bool)
            )

        self.n_output: int = int(bm.sum(self.entry_mask))

    def condense(self, K_local: Any, rho_local: Optional[Any] = None) -> Tuple[Any, Any]:
        """由代理网络预测缩聚刚度矩阵, 失败时回退精确 Schur 补.

        参数:
            K_local: 单个子结构的局部刚度矩阵, 形状 ``(n_dof, n_dof)``. 代理路径
                不使用它预测 ``K_s``, 但恢复矩阵 ``N`` 与回退路径都由它精确导出.
            rho_local: 子结构局部密度场, 展平后作为网络输入. 为 ``None`` 时直接
                回退精确缩聚.

        返回:
            (K_s, N): ``K_s`` 形状 ``(n_b, n_b)``, ``N`` 形状 ``(n_i, n_b)``.
                ``N`` 恒为精确值, 代理只替换 ``K_s``.

        异常:
            SurrogateContractError: 网络输出维与参数化所需的独立条目数不符时抛出.
                其余异常一律转为精确回退.

        说明:
            本实现逐个子结构推理, 不接受前导批量维: 网络输入维固定为单个子结构
            的密度分量数. 调用后可读 ``used_fallback`` 判断本次是否发生回退.
        """
        self.used_fallback = False

        # 若未提供模型或材料密度输入, 直接触发精确回退.
        if self.model is None or rho_local is None:
            self.used_fallback = True
            self.K_s, self.N = self.fallback_solver.condense(K_local, rho_local)
            return self.K_s, self.N

        try:
            self.model.eval()
            with torch.no_grad():
                rho_np = bm.to_numpy(rho_local).flatten()
                x_tensor = torch.tensor(rho_np, dtype=torch.float32).unsqueeze(0)
                pred_vec_np = self.model(x_tensor).squeeze(0).cpu().numpy()

            pred_vec = bm.asarray(pred_vec_np, dtype=bm.float64)
            if not bool(bm.all(bm.isfinite(pred_vec))):
                raise ValueError("代理网络输出含 NaN 或 Inf")
            if pred_vec.shape[0] != self.n_output:
                raise SurrogateContractError(
                    f"代理网络输出维 {pred_vec.shape[0]} 与参数化所需的 "
                    f"{self.n_output} 个独立条目不符"
                )

            mat = bm.zeros((self.n_reduced, self.n_reduced), dtype=bm.float64)
            mat = bm.set_at(mat, self.entry_mask, pred_vec)

            # 使用索引提取和写回对角线, 避免依赖各后端未统一声明的 bm.diag 接口.
            diag_indices = bm.arange(self.n_reduced)
            if self.is_cholesky:
                # 对角线取绝对值使 L 的对角为正; 不再叠加固定下界, 退化情形交由
                # 门禁拦截, 从而不给完美预测引入额外偏差.
                diag_v = bm.abs(mat[diag_indices, diag_indices])
                L = bm.set_at(mat, (diag_indices, diag_indices), diag_v)

                # 对角线近乎退化意味着 L 的列近乎线性相关, 重构出的算子在变形子
                # 空间上接近奇异. 该判据与量纲无关, 只看对角线内部的相对尺度.
                d_max = float(bm.max(diag_v))
                if d_max <= 0.0 or float(bm.min(diag_v)) <= self.rcond_min * d_max:
                    raise ValueError("预测的 Cholesky 因子对角线数值退化")

                # 投影回全部接口自由度: 刚体子空间上恒为零, 秩亏由构造保证.
                reduced = L @ L.T
                K_s_pred = self.range_basis @ reduced @ self.range_basis.T
            else:
                # mat 存储上三角独立条目. 镜像非对角项并仅保留一次对角项, 得到对称矩阵.
                K_s_pred = mat + mat.T
                K_s_pred = bm.set_at(
                    K_s_pred,
                    (diag_indices, diag_indices),
                    mat[diag_indices, diag_indices],
                )
                # 对称参数化没有任何结构保证, 正定性只能在此判定.
                evals = bm.linalg.eigvalsh(K_s_pred)  # type: ignore[attr-defined]
                if float(bm.to_numpy(evals)[0]) <= 1e-8:
                    raise ValueError("预测的 K_s 阵未满足正定物理条件")

            # 从 K_local 提取精确形函数 N 用于内部位移恢复.
            _, self.N = self.fallback_solver.condense(K_local, rho_local)
            self.K_s = K_s_pred

        except SurrogateContractError:
            # 契约错误属于配置问题, 回退会把它伪装成精度损失, 因此直接上抛.
            raise
        except Exception:
            # 预测异常或未通过门禁时触发精确回退.
            self.used_fallback = True
            self.K_s, self.N = self.fallback_solver.condense(K_local, rho_local)

        return self.K_s, self.N
