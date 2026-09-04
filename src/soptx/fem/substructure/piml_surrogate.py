"""基于神经网络代理的子结构静力缩聚.

提供两条在线构造缩聚刚度的代理路径, 分工相反:

- ``PIMLStaticCondensation`` 直接预测 ``K_s``, 内部位移恢复矩阵 ``N`` 恒由局部刚度
  精确导出;
- ``ShapeFunctionCondensation`` 预测形函数 ``N``, 再由 Huang 2023 式 (17)
  ``K_s = N_full^T K_local N_full`` 导出缩聚刚度. 该式的误差是形函数误差的**二阶**
  量, 但恢复出的内部位移不再精确, 因此解层评判标准严于前者.

两条路径都把预测限制在**变形子空间**上, 使自由漂浮子结构 ``K_s`` 的刚体零空间成为
构造性质而非拟合结果; 预测异常或未通过门禁时一律回退精确 Schur 补, 并置
``used_fallback``. 数学背景, 门禁判据的推导与实测证据见
``docs/fem/substructure-condensation-implementation.md``.

所有数组与代数运算基于 FEALPy 后端管理器 (bm).
"""

from typing import Tuple, Any, Optional
import torch
import torch.nn as nn
from fealpy.backend import backend_manager as bm

from soptx.ml import MLP, PIMLSurrogateNet, ShapeFunctionSurrogateNet

from .condensation import StaticCondensationBase, FEAStaticCondensation


class SurrogateContractError(RuntimeError):
    """代理网络与参数化的接口契约不符.

    与数值退化不同, 该错误表示配置本身有误 (例如网络输出维与独立条目数不匹配),
    静默回退只会把它掩盖成一次精度下降, 因此不参与回退, 直接向上抛出.
    """


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


class ShapeFunctionCondensation(StaticCondensationBase):
    """由代理形函数经 Huang 2023 式 (17) 构造缩聚刚度的子结构缩聚器.

    与 ``PIMLStaticCondensation`` 分工相反: 后者预测 ``K_s`` 而 ``N`` 恒取精确值,
    本类预测 ``N`` 而 ``K_s`` 由

        ``K_s = N_full^T K_local N_full``

    导出, 其中 ``N_full`` 在接口自由度行上为单位阵, 在内部自由度行上为 ``N``. 因此
    本路径的解层误差同时包含形函数误差对内部位移恢复的直接影响, 评判标准严于直接
    预测路径.

    形函数按 ``N = Phi_i R_rigid^T + M R_perp^T`` 参数化, 网络只输出变形子空间上的
    ``M``, 形状 ``(n_i, m)``. 刚体分量 ``Phi_i`` 与密度无关, 由网格解析给出 (见
    ``SubstructurePrototype.rigid_interior_modes``), 不进入网络输出.

    该参数化连同式 (17) 给出三条构造性质, 因此**不需要**对应的门禁:

    1. 记 ``N = N* + E``, 则 ``K_s(N) - K_s(N*) = E^T K_ii E``: 一阶项被
       ``K_ii N* = -K_ib`` 精确抵消, 故式 (17) 的误差是 ``E`` 的二阶量. 又 ``K_ii``
       正定, 该误差半正定, 即式 (17) 只会高估刚度, 与变分原理一致, 由此结构柔度只
       会被低估. 正定性无需判定.
    2. ``K_s`` 限制到变形子空间上对**任意** ``N`` 都严格正定: 若 ``v`` 属于该子空间
       且 ``v^T K_s v = 0``, 则 ``N_full v`` 落在 ``K_local`` 的零空间即刚体模态中,
       其接口分量 ``v`` 因而属于刚体子空间, 与 ``v`` 正交于刚体子空间矛盾. 秩亏结构
       是构造性质而非拟合结果.
    3. ``K_s R_rigid = 0`` 精确成立: 参数化保证 ``N R_rigid = Phi_i``, 于是
       ``N_full R_rigid`` 恰是子结构的刚体位移场, 被 ``K_local`` 零化.

    真正没有保证的是**高估的幅度**: ``E`` 任意大时 ``E^T K_ii E`` 也任意大. 门禁据此
    分三道, 按代价递增排列:

    - **刚体零空间残量**: ``|K_s R_rigid| / |K_s|`` 应在舍入量级. 它复核上述性质 3,
      用于拦截基, ``Phi_i`` 或形状本身的不一致, 代价仅一次矩阵乘法.
    - **刚化上界**: ``lambda_max(K_s - K_bb) <= excess_rtol * lambda_max(K_bb)``.
      ``K_bb`` 是内部自由度完全固支时的接口刚度, 即式 (17) 取 ``N = 0`` 的结果, 可由
      ``K_local`` 直接切片得到而无需任何分解. 精确形函数下
      ``K_s - K_bb = -K_bi K_ii^{-1} K_ib`` 半负定, 该判据严格满足且有余量; 预测形
      函数下超出部分恰是 ``E^T K_ii E`` 在最坏方向上的刚化能量. 这是唯一一道实质
      门禁, 且方向与性质 1 允许的误差方向一致.
    - **变形子空间条件数**: ``lambda_min / lambda_max >= rcond_min``. 性质 2 只保证
      定性非零, 比值仍可能退化到使全局接口系统病态, 故按与 ``PIMLStaticCondensation``
      相同的口径复核.

    每次调用后 ``gate_report`` 记录上述三个实测比值, 供门禁标定与证据留存; 回退时
    该字典保留触发回退的那次预测的读数.

    阈值缺省值由 ``examples/piml_substructure_elasticity`` 的 5x5 子结构标定, 换算例
    后应重新标定并重跑该脚本的步骤 5:

    - ``rigid_tol`` 与 ``rcond_min`` 对应构造性质, 实测读数分别为 ``1.2e-16`` 与
      ``6.1e-3``, 距阈值有五个数量级以上, 仅用于拦截配置错误;
    - ``excess_rtol`` 是唯一会因精度不足而触发的门禁. 留出集上刚化幅度最大
      ``8.1e-3``, 相对阈值仅 ``2.5`` 倍裕度——这是有意的: ``2e-2`` 对应形函数误差约
      ``22%``, 正是本算例中式 (17) 路径与直接预测 ``K_s`` 路径精度相当的交叉点, 越过
      它本路径不再有优势, 回退精确缩聚反而更合算.

    门禁代价为 ``O(n_b^3)`` (两次接口尺度的特征值分解), 而精确缩聚代价随内部自由度
    按 ``O(n_i^3)`` 增长. 内部自由度按体积增长而接口自由度按表面增长, 因此子结构越大
    门禁越便宜: 实测 5x5 时门禁是一次精确缩聚的 ``5.6`` 倍, 12x12 时降到 ``0.55`` 倍,
    20x20 时降到 ``0.34`` 倍.
    """

    def __init__(
        self,
        i_dofs: Any,
        b_dofs: Any,
        model: Optional[nn.Module] = None,
        rigid_basis: Optional[Any] = None,
        deformation_basis: Optional[Any] = None,
        rigid_interior: Optional[Any] = None,
        rigid_tol: float = 1.0e-10,
        excess_rtol: float = 2.0e-2,
        rcond_min: float = 1.0e-8,
    ) -> None:
        """初始化形函数路径缩聚器.

        参数:
            i_dofs: 子结构内部自由度的局部编号.
            b_dofs: 子结构接口自由度的局部编号.
            model: 预测变形子空间分量 ``M`` 的代理网络. 为 ``None`` 时总是回退精确
                缩聚.
            rigid_basis: 接口刚体模态的标准正交基 ``R_rigid``, 形状
                ``(n_b, n_rigid)``, 取 ``SubstructurePrototype.rigid_basis``.
            deformation_basis: 变形子空间的标准正交基 ``R_perp``, 形状 ``(n_b, m)``,
                取 ``SubstructurePrototype.deformation_basis``.
            rigid_interior: 刚体运动下内部自由度的取值 ``Phi_i``, 形状
                ``(n_i, n_rigid)``, 取 ``SubstructurePrototype.rigid_interior_modes``.
                三者必须来自同一个原型, 否则 ``N R_rigid = Phi_i`` 不成立.
            rigid_tol: 刚体零空间残量的相对上限. 该量由构造保持在舍入量级, 阈值
                只需远高于舍入而远低于任何真实的不一致.
            excess_rtol: 相对 ``K_bb`` 的刚化上界, 即允许的最大相对高估幅度. 三道
                门禁中唯一与精度有关的一道, 换算例后须重新标定.
            rcond_min: 变形子空间上最小与最大特征值之比的下限.

        异常:
            ValueError: 三组基未全部给出, 或其形状与 ``n_i``, ``n_b`` 不符时抛出.
        """
        super().__init__(i_dofs, b_dofs)
        self.model = model
        self.rigid_tol = float(rigid_tol)
        self.excess_rtol = float(excess_rtol)
        self.rcond_min = float(rcond_min)

        # 精确缩聚器是模型缺失, 预测失败或门禁失败时的回退路径.
        self.fallback_solver = FEAStaticCondensation(i_dofs, b_dofs)
        self.used_fallback = False
        self.gate_report: dict = {}

        missing = [
            name for name, value in (
                ("rigid_basis", rigid_basis),
                ("deformation_basis", deformation_basis),
                ("rigid_interior", rigid_interior),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                f"形函数参数化需要给出 {', '.join(missing)}; "
                f"三者均可由 SubstructurePrototype 的同名属性取得."
            )

        self.rigid_basis: Any = bm.asarray(rigid_basis, dtype=bm.float64)
        self.deformation_basis: Any = bm.asarray(deformation_basis, dtype=bm.float64)
        self.rigid_interior: Any = bm.asarray(rigid_interior, dtype=bm.float64)

        self.n_rigid: int = int(self.rigid_basis.shape[1])
        self.n_reduced: int = int(self.deformation_basis.shape[1])

        expected = {
            "rigid_basis": (self.rigid_basis, (self.n_b, self.n_rigid)),
            "deformation_basis": (self.deformation_basis, (self.n_b, self.n_reduced)),
            "rigid_interior": (self.rigid_interior, (self.n_i, self.n_rigid)),
        }
        for name, (value, shape) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} 的形状应为 {shape}; 当前为 {tuple(value.shape)}."
                )
        if self.n_rigid + self.n_reduced != self.n_b:
            raise ValueError(
                f"刚体子空间与变形子空间的维数之和应为接口自由度数 {self.n_b}; "
                f"当前为 {self.n_rigid} + {self.n_reduced}."
            )

        self.n_output: int = self.n_i * self.n_reduced

    def _check_gates(self, K_s_pred: Any, K_bb: Any) -> None:
        """按三道门禁复核预测的缩聚刚度, 并把实测比值写入 ``gate_report``.

        参数:
            K_s_pred: 由式 (17) 得到的缩聚刚度, 形状 ``(n_b, n_b)``.
            K_bb: 局部刚度的接口分块, 形状 ``(n_b, n_b)``, 即内部固支时的接口刚度.

        异常:
            ValueError: 任一门禁未通过时抛出, 由 ``condense`` 转为精确回退.

        说明:
            判据的推导见类文档. 三道门禁按代价递增排列并逐道短路, 使常见的粗差在
            进入特征值分解前就被拦下. 特征值分解只读取矩阵的一个三角, 因此舍入级
            的非对称不影响判定.
        """
        # [1] 刚体零空间残量: 复核参数化的构造性质, 只需一次矩阵乘法.
        rigid_residual = float(
            bm.linalg.norm(K_s_pred @ self.rigid_basis) / bm.linalg.norm(K_s_pred)
        )
        self.gate_report = {"rigid_residual": rigid_residual}
        if not rigid_residual <= self.rigid_tol:
            self.gate_report["failed_gate"] = "rigid_residual"
            self.gate_report["gate_limit"] = self.rigid_tol
            raise ValueError(
                f"预测刚度的刚体零空间残量 {rigid_residual:.3e} 超过 "
                f"{self.rigid_tol:.3e}"
            )

        # [2] 刚化上界: 高估幅度相对于内部固支刚度 K_bb 的最坏方向读数.
        evals_bb = bm.to_numpy(bm.linalg.eigvalsh(K_bb))  # type: ignore[attr-defined]
        evals_excess = bm.to_numpy(
            bm.linalg.eigvalsh(K_s_pred - K_bb)  # type: ignore[attr-defined]
        )
        excess_ratio = float(evals_excess[-1]) / float(evals_bb[-1])
        self.gate_report["excess_ratio"] = excess_ratio
        if not excess_ratio <= self.excess_rtol:
            self.gate_report["failed_gate"] = "excess_ratio"
            self.gate_report["gate_limit"] = self.excess_rtol
            raise ValueError(
                f"预测刚度相对内部固支刚度的刚化幅度 {excess_ratio:.3e} 超过 "
                f"{self.excess_rtol:.3e}"
            )

        # [3] 变形子空间条件数: 非零由构造保证, 此处只拦截比值退化.
        reduced = (
            bm.matrix_transpose(self.deformation_basis)
            @ K_s_pred @ self.deformation_basis
        )
        evals_red = bm.to_numpy(bm.linalg.eigvalsh(reduced))  # type: ignore[attr-defined]
        rcond = float(evals_red[0]) / float(evals_red[-1])
        self.gate_report["reduced_rcond"] = rcond
        if not rcond >= self.rcond_min:
            self.gate_report["failed_gate"] = "reduced_rcond"
            self.gate_report["gate_limit"] = self.rcond_min
            raise ValueError(
                f"预测刚度在变形子空间上的条件数倒数 {rcond:.3e} 低于 "
                f"{self.rcond_min:.3e}"
            )

    def condense(self, K_local: Any, rho_local: Optional[Any] = None) -> Tuple[Any, Any]:
        """预测形函数并按式 (17) 计算缩聚刚度, 未过门禁时回退精确 Schur 补.

        参数:
            K_local: 单个子结构的局部刚度矩阵, 形状 ``(n_dof, n_dof)``. 它既是式
                (17) 的算子, 也是回退路径的输入.
            rho_local: 子结构局部密度场, 展平后作为网络输入. 为 ``None`` 时直接回退
                精确缩聚.

        返回:
            (K_s, N): ``K_s`` 形状 ``(n_b, n_b)``, ``N`` 形状 ``(n_i, n_b)``. 两者
                同为预测量: ``N`` 由网络给出, ``K_s`` 由 ``N`` 经式 (17) 导出.

        异常:
            SurrogateContractError: 网络输出维与 ``n_i * m`` 不符时抛出. 其余异常
                一律转为精确回退.
            ValueError: ``K_local`` 的形状与当前自由度划分不一致时抛出.

        说明:
            本实现逐个子结构推理, 不接受前导批量维. 式 (17) 按分块展开
            ``K_bb + A + A^T + N^T K_ii N`` (``A = K_bi N``) 计算, 与显式构造
            ``N_full`` 再做三重乘积在代数上等价, 但只在 ``(n_b, n_b)`` 上运算,
            且省去一次散射赋值. 调用后可读 ``used_fallback`` 与 ``gate_report``.
        """
        self._check_local_stiffness(K_local)
        self.used_fallback = False

        # 接口分块同时是式 (17) 的常数项和刚化上界门禁的参照量.
        K_bb = K_local[self.b_dofs[:, None], self.b_dofs]

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
                    f"代理网络输出维 {pred_vec.shape[0]} 与形函数参数化所需的 "
                    f"{self.n_output} = n_i * m 个分量不符"
                )

            # 刚体分量由构造给出, 网络只填变形子空间上的分量.
            M = bm.reshape(pred_vec, (self.n_i, self.n_reduced))
            N_pred = (
                self.rigid_interior @ bm.matrix_transpose(self.rigid_basis)
                + M @ bm.matrix_transpose(self.deformation_basis)
            )

            K_ii = K_local[self.i_dofs[:, None], self.i_dofs]
            K_ib = K_local[self.i_dofs[:, None], self.b_dofs]
            A = bm.matrix_transpose(K_ib) @ N_pred
            K_s_pred = (
                K_bb + A + bm.matrix_transpose(A)
                + bm.matrix_transpose(N_pred) @ K_ii @ N_pred
            )

            self._check_gates(K_s_pred, K_bb)

            self.N = N_pred
            self.K_s = K_s_pred

        except SurrogateContractError:
            # 契约错误属于配置问题, 回退会把它伪装成精度损失, 因此直接上抛.
            raise
        except Exception:
            # 预测异常或未通过门禁时触发精确回退.
            self.used_fallback = True
            self.K_s, self.N = self.fallback_solver.condense(K_local, rho_local)

        return self.K_s, self.N
