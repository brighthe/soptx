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
    """由代理内部延拓经 Huang 2023 式 (17) 构造降阶刚度的子结构缩聚器.

    与 ``PIMLStaticCondensation`` 分工相反: 后者预测 ``K_s`` 而 ``N`` 恒取精确值,
    本类预测内部延拓 ``B`` 而降阶刚度 ``K_r`` 由

        ``K_r = (H^T)^T K_local H^T``,    ``H^T = [B; T]``

    导出, 其中 ``H^T`` 在接口自由度行上为迹矩阵 ``T``, 在内部自由度行上为 ``B``.
    因此本路径的解层误差同时包含延拓误差对内部位移恢复的直接影响, 评判标准严于
    直接预测路径.

    ``T`` 由构造参数 ``trace`` 给出, 形状 ``(n_b, n_q)``, 迹坐标与完整接口位移满足
    ``u_b = T q``; ``trace=None`` 取完整接口 ``T = I``, 此时 ``B`` 退化为完整接口
    形函数 ``N``, ``K_r`` 退化为 Schur 补 ``K_s``. 迹空间不是事后投影: 网络的输出
    维数 ``n_i * m`` 定义在迹空间上, ``m = n_q - n_rigid``, 角点线性迹下它比完整
    接口小一个量级. 因此本类返回的 ``(K_r, B)`` 已在迹空间中, 与
    ``LocalReductionResult`` 对精确缩聚"迹投影属于后续分析阶段"的约定不同, 这是
    该参数化的固有性质.

    内部延拓按 ``B = Phi_i R_q^T + M R_perp^T`` 参数化, 网络只输出变形子空间上的
    ``M``, 形状 ``(n_i, m)``. 刚体分量 ``Phi_i`` 与密度无关, 由网格解析给出 (见
    ``SubstructurePrototype.trace_interface_bases``), 不进入网络输出.

    该参数化连同式 (17) 给出三条构造性质, 因此**不需要**对应的门禁:

    1. 记 ``B = B* + E``, 则 ``K_r(B) - K_r(B*) = E^T K_ii E``: 一阶项被
       ``K_ii B* = -K_ib T`` 精确抵消, 故变分式的误差是 ``E`` 的二阶量. 又 ``K_ii``
       正定, 该误差半正定, 即变分式只会高估刚度, 与变分原理一致, 由此结构柔度只
       会被低估. 正定性无需判定.
    2. ``K_r`` 限制到变形子空间上对**任意** ``B`` 都严格正定: 若 ``v`` 属于该子空间
       且 ``v^T K_r v = 0``, 则 ``H^T v`` 落在 ``K_local`` 的零空间即刚体模态中,
       其迹分量 ``v`` 因而属于刚体子空间, 与 ``v`` 正交于刚体子空间矛盾. 秩亏结构
       是构造性质而非拟合结果. 该论证要求 ``T`` 列满秩且迹空间能表示刚体运动, 两者
       由 ``trace_interface_bases`` 校验.
    3. ``K_r R_q = 0`` 精确成立: 参数化保证 ``B R_q = Phi_i``, 于是 ``H^T R_q``
       恰是子结构的刚体位移场, 被 ``K_local`` 零化.

    真正没有保证的是**高估的幅度**: ``E`` 任意大时 ``E^T K_ii E`` 也任意大. 门禁据此
    分三道, 按代价递增排列:

    - **刚体零空间残量**: ``|K_s R_rigid| / |K_s|`` 应在舍入量级. 它复核上述性质 3,
      用于拦截基, ``Phi_i`` 或形状本身的不一致, 代价仅一次矩阵乘法.
    - **刚化上界**: 记 ``K_bb^T = T^T K_bb T``, 判据为
      ``lambda_max(K_r - K_bb^T) <= excess_rtol * lambda_max(K_bb^T)``. ``K_bb^T``
      是内部自由度完全固支时的迹空间接口刚度, 即变分式取 ``B = 0`` 的结果, 由
      ``K_local`` 切片后一次投影得到而无需任何分解. 精确延拓下
      ``K_r - K_bb^T = -(K_ib T)^T K_ii^{-1} (K_ib T)`` 半负定, 该判据严格满足且有
      余量; 预测延拓下超出部分恰是 ``E^T K_ii E`` 在最坏方向上的刚化能量. 这是唯一
      一道实质门禁, 且方向与性质 1 允许的误差方向一致.
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

    门禁代价为 ``O(n_q^3)`` (两次迹空间尺度的特征值分解), 而精确缩聚代价随内部自由度
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
        trace: Optional[Any] = None,
    ) -> None:
        """初始化形函数路径缩聚器.

        参数:
            i_dofs: 子结构内部自由度的局部编号.
            b_dofs: 子结构接口自由度的局部编号.
            model: 预测变形子空间分量 ``M`` 的代理网络. 为 ``None`` 时总是回退精确
                缩聚.
            rigid_basis: 迹空间刚体模态的标准正交基 ``R_q``, 形状
                ``(n_q, n_rigid)``, 取 ``SubstructurePrototype.trace_interface_bases``
                的第一个返回值.
            deformation_basis: 变形子空间的标准正交基 ``R_perp``, 形状
                ``(n_q, m)``, 取上述方法的第二个返回值.
            rigid_interior: 刚体运动下内部自由度的取值 ``Phi_i``, 形状
                ``(n_i, n_rigid)``, 取上述方法的第三个返回值. 三者必须来自同一个
                原型与同一个 ``trace``, 否则 ``B R_q = Phi_i`` 不成立.
            rigid_tol: 刚体零空间残量的相对上限. 该量由构造保持在舍入量级, 阈值
                只需远高于舍入而远低于任何真实的不一致.
            excess_rtol: 相对 ``K_bb`` 的刚化上界, 即允许的最大相对高估幅度. 三道
                门禁中唯一与精度有关的一道, 换算例后须重新标定.
            rcond_min: 变形子空间上最小与最大特征值之比的下限.
            trace: 接口迹空间 ``TraceBasis``, 提供形状 ``(n_b, n_q)`` 的迹矩阵
                ``T``. 为 ``None`` 时取完整接口 ``T = I``, 此时 ``n_q == n_b``,
                行为与本参数引入前逐位一致.

        异常:
            ValueError: 三组基未全部给出, 其形状与 ``n_i``, ``n_q`` 不符, 或
                ``trace`` 的接口自由度数与 ``b_dofs`` 不符时抛出.
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

        self.trace = trace
        if trace is None:
            self.trace_matrix: Optional[Any] = None
            self.n_trace: int = self.n_b
        else:
            self.trace_matrix = bm.asarray(trace.matrix, dtype=bm.float64)
            if int(self.trace_matrix.shape[0]) != self.n_b:
                raise ValueError(
                    f"trace 的完整接口自由度数应为 {self.n_b}; "
                    f"当前迹矩阵形状为 {tuple(self.trace_matrix.shape)}."
                )
            self.n_trace = int(self.trace_matrix.shape[1])

        self.rigid_basis: Any = bm.asarray(rigid_basis, dtype=bm.float64)
        self.deformation_basis: Any = bm.asarray(deformation_basis, dtype=bm.float64)
        self.rigid_interior: Any = bm.asarray(rigid_interior, dtype=bm.float64)

        self.n_rigid: int = int(self.rigid_basis.shape[1])
        self.n_reduced: int = int(self.deformation_basis.shape[1])

        expected = {
            "rigid_basis": (self.rigid_basis, (self.n_trace, self.n_rigid)),
            "deformation_basis": (
                self.deformation_basis, (self.n_trace, self.n_reduced)
            ),
            "rigid_interior": (self.rigid_interior, (self.n_i, self.n_rigid)),
        }
        for name, (value, shape) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} 的形状应为 {shape}; 当前为 {tuple(value.shape)}."
                )
        if self.n_rigid + self.n_reduced != self.n_trace:
            raise ValueError(
                f"刚体子空间与变形子空间的维数之和应为迹自由度数 {self.n_trace}; "
                f"当前为 {self.n_rigid} + {self.n_reduced}."
            )

        self.n_output: int = self.n_i * self.n_reduced

    def _trace_blocks(self, K_local: Any) -> Tuple[Any, Any, Any]:
        """把局部刚度的分块映射到当前迹空间.

        参数:
            K_local: 局部刚度矩阵, 形状 ``(..., n_dof, n_dof)``, 允许前导批量维.

        返回:
            (K_ii, K_ib_t, K_bb_t): 形状分别为 ``(..., n_i, n_i)``,
                ``(..., n_i, n_q)`` 与 ``(..., n_q, n_q)``. ``T = I`` 时后两者即
                原始的 ``K_ib`` 与 ``K_bb``.
        """
        K_ii = K_local[..., self.i_dofs[:, None], self.i_dofs]
        K_ib = K_local[..., self.i_dofs[:, None], self.b_dofs]
        K_bb = K_local[..., self.b_dofs[:, None], self.b_dofs]
        if self.trace_matrix is None:
            return K_ii, K_ib, K_bb
        T = self.trace_matrix
        return K_ii, K_ib @ T, bm.matrix_transpose(T) @ K_bb @ T

    def _reduce_exact(self, exact: Tuple[Any, Any]) -> Tuple[Any, Any]:
        """把完整接口的精确缩聚结果降到当前迹空间.

        参数:
            exact: ``fallback_solver.condense`` 的返回值 ``(K_s, N)``, 形状分别
                为 ``(n_b, n_b)`` 与 ``(n_i, n_b)``.

        返回:
            (K_r, B): ``K_r = T^T K_s T``, ``B = N T``. ``T = I`` 时原样返回.
        """
        K_s, N = exact
        if self.trace_matrix is None:
            return K_s, N
        T = self.trace_matrix
        return bm.matrix_transpose(T) @ K_s @ T, N @ T

    def assemble_recovery(self, deformation: Any) -> Any:
        """由变形子空间分量 ``M`` 合成迹空间下的内部延拓 ``B``.

        参数:
            deformation: 网络输出的 ``M``, 形状 ``(..., n_i, m)``, 允许前导
                批量维.

        返回:
            ``B = Phi_i R_q^T + M R_perp^T``, 形状 ``(..., n_i, n_q)``. 由
            ``R_q`` 与 ``R_perp`` 正交可知 ``B R_q = Phi_i`` 恒成立, 刚体分量
            不受 ``M`` 影响.
        """
        return (
            self.rigid_interior @ bm.matrix_transpose(self.rigid_basis)
            + deformation @ bm.matrix_transpose(self.deformation_basis)
        )

    def project_deformation(self, B: Any) -> Any:
        """把内部延拓投影到变形子空间, 取出网络的训练目标 ``M``.

        参数:
            B: 迹空间下的内部延拓, 形状 ``(..., n_i, n_q)``.

        返回:
            ``M = B R_perp``, 形状 ``(..., n_i, m)``. 它是
            ``assemble_recovery`` 的左逆: 对满足 ``B R_q = Phi_i`` 的 ``B``,
            两者复合还原 ``B``.
        """
        return B @ self.deformation_basis

    def assemble_reduced_stiffness(self, K_local: Any, B: Any) -> Any:
        """由给定的内部延拓按变分式构造迹空间降阶刚度.

        参数:
            K_local: 局部刚度矩阵, 形状 ``(..., n_dof, n_dof)``.
            B: 迹空间下的内部延拓, 形状 ``(..., n_i, n_q)``.

        返回:
            ``K_r``, 形状 ``(..., n_q, n_q)``.

        说明:
            本方法是变分构造 ``K_r = (H^T)^T K_local H^T`` 的唯一实现,
            ``condense`` 与 ``PIMLShapeReduction.reduce_many`` 都经由它,
            精度评估脚本亦应直接调用而不另写一份. 传入精确的 ``B`` 得到精确的
            ``K_r``, 传入预测的 ``B`` 得到预测的 ``K_r``, 两者之差即类文档性质
            1 中的 ``E^T K_ii E``.
        """
        K_ii, K_ib_t, K_bb_t = self._trace_blocks(K_local)
        return self._variational_stiffness(K_ii, K_ib_t, K_bb_t, B)

    @staticmethod
    def _variational_stiffness(
        K_ii: Any, K_ib_t: Any, K_bb_t: Any, B: Any
    ) -> Any:
        """在已切片的迹空间分块上展开变分式.

        参数:
            K_ii, K_ib_t, K_bb_t: ``_trace_blocks`` 的三个返回值.
            B: 迹空间下的内部延拓.

        返回:
            ``T^T K_bb T + A + A^T + B^T K_ii B``, 其中 ``A = (K_ib T)^T B``.
            调用方多半已持有 ``K_bb_t`` (刚化上界门禁的参照量), 故与
            ``assemble_reduced_stiffness`` 分开, 避免重复切片.
        """
        A = bm.matrix_transpose(K_ib_t) @ B
        return (
            K_bb_t + A + bm.matrix_transpose(A)
            + bm.matrix_transpose(B) @ K_ii @ B
        )

    def _check_gates(self, K_s_pred: Any, K_bb: Any) -> None:
        """按三道门禁复核预测的缩聚刚度, 并把实测比值写入 ``gate_report``.

        参数:
            K_s_pred: 由变分式得到的降阶刚度, 形状 ``(n_q, n_q)``.
            K_bb: 迹空间接口分块 ``T^T K_bb T``, 形状 ``(n_q, n_q)``, 即内部固支
                时的迹空间接口刚度.

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
        """预测内部延拓并按变分式计算降阶刚度, 未过门禁时回退精确 Schur 补.

        参数:
            K_local: 单个子结构的局部刚度矩阵, 形状 ``(n_dof, n_dof)``. 它既是式
                (17) 的算子, 也是回退路径的输入.
            rho_local: 子结构局部密度场, 展平后作为网络输入. 为 ``None`` 时直接回退
                精确缩聚.

        返回:
            (K_r, B): ``K_r`` 形状 ``(n_q, n_q)``, ``B`` 形状 ``(n_i, n_q)``. 两者
                同为预测量: ``B`` 由网络给出, ``K_r`` 由 ``B`` 经变分式导出.
                ``T = I`` 时 ``n_q == n_b``, 两者即完整接口上的 ``K_s`` 与 ``N``.

        异常:
            SurrogateContractError: 网络输出维与 ``n_i * m`` 不符时抛出. 其余异常
                一律转为精确回退.
            ValueError: ``K_local`` 的形状与当前自由度划分不一致时抛出.

        说明:
            本实现逐个子结构推理, 不接受前导批量维. 变分式按分块展开
            ``T^T K_bb T + A + A^T + B^T K_ii B`` (``A = (K_ib T)^T B``) 计算,
            与显式构造 ``H^T = [B; T]`` 再做三重乘积 ``(H^T)^T K H^T`` 在代数上
            等价, 但只在 ``(n_q, n_q)`` 上运算, 且省去一次散射赋值. ``T = I``
            时逐项退化为 ``K_bb + A + A^T + N^T K_ii N``. 调用后可读
            ``used_fallback`` 与 ``gate_report``.
        """
        self._check_local_stiffness(K_local)
        self.used_fallback = False

        # 迹空间接口分块同时是变分构造的常数项和刚化上界门禁的参照量.
        K_ii, K_ib_t, K_bb_t = self._trace_blocks(K_local)

        if self.model is None or rho_local is None:
            self.used_fallback = True
            self.K_s, self.N = self._reduce_exact(
                self.fallback_solver.condense(K_local, rho_local)
            )
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
            B_pred = self.assemble_recovery(M)
            K_r_pred = self._variational_stiffness(K_ii, K_ib_t, K_bb_t, B_pred)

            self._check_gates(K_r_pred, K_bb_t)

            self.N = B_pred
            self.K_s = K_r_pred

        except SurrogateContractError:
            # 契约错误属于配置问题, 回退会把它伪装成精度损失, 因此直接上抛.
            raise
        except Exception:
            # 预测异常或未通过门禁时触发精确回退.
            self.used_fallback = True
            self.K_s, self.N = self._reduce_exact(
                self.fallback_solver.condense(K_local, rho_local)
            )

        return self.K_s, self.N
