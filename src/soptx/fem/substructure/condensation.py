"""子结构静力缩聚.

提供基于 FEALPy ``bm`` 后端的精确 Schur 补缩聚实现. 所有缩聚接口都接受任意
可变前导维 ``...``, 单个子结构只是前导维为空的特例, 批量子结构对应前导维 ``B``.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional, Dict
import numpy as np
from fealpy.backend import backend_manager as bm

from .traces import LinearCornerTraceBasis, TraceBasis


class StaticCondensationBase(ABC):
    """
    子结构静力缩聚抽象基类.

    接口自由度 (b) 与内部自由度 (i) 的局部刚度矩阵分块关系为:
        K_local = [[K_ii, K_ib],
                   [K_bi, K_bb]]
        u_i = N * u_b, 其中 N = - inv(K_ii) * K_ib
        K_s = K_bb - K_bi * inv(K_ii) * K_ib

    子类通过 ``condense`` 计算或预测缩聚刚度矩阵 ``K_s`` 与内部位移恢复矩阵
    ``N``. 全局接口问题求解完成后, ``recover`` 使用 ``N`` 将接口位移 ``u_b``
    恢复为内部位移 ``u_i``.

    ``i_dofs`` 与 ``b_dofs`` 必须使用同一局部自由度编号, 互不重叠, 并共同构成
    局部刚度矩阵自由度的完整划分. 该条件在构造时一次性校验, 缩聚热路径不再
    重复检查.

    本类的所有形状契约都以 ``...`` 表示可变前导维: ``K_local`` 为
    ``(..., n_dof, n_dof)``, ``K_s`` 为 ``(..., n_b, n_b)``, ``N`` 为
    ``(..., n_i, n_b)``. 前导维为空即单个子结构, 前导维为 ``B`` 即同构子结构
    批次. 实现不对输入做 dtype 转换, 计算精度由调用方传入的 ``K_local`` 决定.
    """

    def __init__(self, i_dofs: Any, b_dofs: Any) -> None:
        """
        初始化子结构自由度划分.

        参数:
            i_dofs: 局部刚度矩阵中内部自由度的索引, 形状 ``(n_i,)``. 这些自由度
                在缩聚时被消去.
            b_dofs: 局部刚度矩阵中边界或接口自由度的索引, 形状 ``(n_b,)``.
                缩聚后的 ``K_s`` 以这些自由度为行列顺序.

        异常:
            ValueError: 当 ``i_dofs`` 与 ``b_dofs`` 的并集不是 ``[0, n_i + n_b)``
                的一个排列时抛出, 即存在负索引, 重复索引, 两者重叠或未覆盖全部
                局部自由度.

        说明:
            输入索引会转换为 ``int64`` 张量. 局部自由度总数由 ``n_i + n_b`` 推出,
            无需 ``K_local`` 参与校验. ``K_s`` 和 ``N`` 在调用 ``condense`` 前
            为 ``None``.
        """
        # 统一自由度索引的后端与数据类型, 以支持后续高级索引.
        self.i_dofs: Any = bm.asarray(i_dofs, dtype=bm.int64)
        self.b_dofs: Any = bm.asarray(b_dofs, dtype=bm.int64)
        self.n_i: int = len(self.i_dofs)
        self.n_b: int = len(self.b_dofs)
        self.n_dofs: int = self.n_i + self.n_b

        # 排序后与 arange 逐位相等, 等价于同时保证非负, 无重复, 不重叠且完整覆盖;
        # 校验只在构造时执行一次, 避免污染每次迭代都会调用的缩聚热路径.
        partition = bm.sort(bm.concat([self.i_dofs, self.b_dofs]))
        if not bool(bm.all(partition == bm.arange(self.n_dofs, dtype=bm.int64))):
            raise ValueError(
                f"i_dofs 与 b_dofs 必须构成 [0, {self.n_dofs}) 的完整划分: "
                f"要求非负, 无重复, 互不重叠且覆盖全部局部自由度; "
                f"当前 n_i={self.n_i}, n_b={self.n_b}."
            )

        # 由 condense 写入的缩聚刚度矩阵和内部位移恢复矩阵.
        self.K_s: Optional[Any] = None
        self.N: Optional[Any] = None

    def _check_local_stiffness(self, K_local: Any) -> None:
        """校验局部刚度矩阵的形状与当前自由度划分一致.

        参数:
            K_local: 待校验的局部刚度矩阵, 形状应为 ``(..., n_dof, n_dof)``.

        异常:
            ValueError: 当维数低于 2, 最后两维不是方阵, 或方阵阶数不等于
                ``n_i + n_b`` 时抛出.
        """
        if K_local.ndim < 2:
            raise ValueError(
                f"K_local 至少需要二维, 形状为 (..., n_dof, n_dof); "
                f"当前 ndim={K_local.ndim}."
            )
        if K_local.shape[-1] != K_local.shape[-2]:
            raise ValueError(
                f"K_local 的最后两维必须构成方阵; 当前形状为 {tuple(K_local.shape)}."
            )
        if K_local.shape[-1] != self.n_dofs:
            raise ValueError(
                f"K_local 的局部自由度数为 {K_local.shape[-1]}, "
                f"与 i_dofs 和 b_dofs 给出的 {self.n_dofs} 不一致."
            )

    @abstractmethod
    def condense(self, K_local: Any, rho_local: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        计算或预测缩聚刚度矩阵 ``K_s`` 及内部位移恢复矩阵 ``N``.

        参数:
            K_local: 按当前子结构局部自由度编号排列的局部刚度矩阵, 形状
                ``(..., n_dof, n_dof)``. 前导维为空表示单个子结构, 前导维 ``B``
                表示同构子结构批次.
            rho_local: 子结构局部密度场, 前导维需与 ``K_local`` 一致. 精确有限元
                缩聚可忽略该参数, 代理缩聚器可将其用作预测 ``K_s`` 和 ``N`` 的输入.

        返回:
            (K_s, N): ``K_s`` 形状为 ``(..., n_b, n_b)``, ``N`` 形状为
                ``(..., n_i, n_b)``.
        """
        raise NotImplementedError("子类必须实现 condense() 方法.")

    def recover(self, u_b: Any) -> Any:
        """
        根据接口位移恢复内部位移.

        参数:
            u_b: 按 ``b_dofs`` 顺序排列的接口位移, 形状 ``(..., n_b)``. 其前导维
                与 ``N`` 的前导维按广播规则对齐, 因此单个接口位移向量可与批量
                ``N`` 组合使用.

        返回:
            u_i: 按 ``i_dofs`` 顺序排列的内部位移 ``u_i = N @ u_b``, 形状
                ``(..., n_i)``.

        异常:
            RuntimeError: 当尚未调用 ``condense`` 而恢复矩阵 ``N`` 未生成时抛出.

        说明:
            批量情形下 ``N`` 的末两维才是矩阵维, 因此使用 ``bm.einsum`` 而非 ``@``
            表达矩阵-向量积. 实现不转换 ``u_b`` 的 dtype, 混合精度由调用方负责.
        """
        if self.N is None:
            raise RuntimeError("必须先调用 condense() 方法才能执行 recover().")
        u_b_bm = bm.asarray(u_b)
        return bm.einsum('...ij, ...j -> ...i', self.N, u_b_bm)


class FEAStaticCondensation(StaticCondensationBase):
    """
    有限元精确 Schur 补静态缩聚器.

    该实现显式消去子结构内部自由度, 为 PIML 代理缩聚和全局接口装配提供
    精确有限元基线. 单个子结构与同构子结构批次共用同一份实现.
    """

    def condense(self, K_local: Any, rho_local: Optional[Any] = None) -> Tuple[Any, Any]:
        """计算精确 Schur 补缩聚刚度矩阵与内部位移恢复矩阵.

        参数:
            K_local: 局部刚度矩阵, 形状 ``(..., n_dof, n_dof)``. 其行列编号必须与
                ``i_dofs`` 和 ``b_dofs`` 使用的局部自由度编号一致, 无需重排为内部
                自由度在前, 接口自由度在后的块顺序. 末两维必须对称.
            rho_local: 为与代理缩聚器保持统一方法签名而保留. 精确 Schur 补不使用
                该参数.

        返回:
            (K_s, N): 形状 ``(..., n_b, n_b)`` 的缩聚刚度矩阵和形状
                ``(..., n_i, n_b)`` 的内部位移恢复矩阵.

        异常:
            ValueError: 当 ``K_local`` 的形状与当前自由度划分不一致时抛出.

        说明:
            内部刚度块 ``K_ii`` 必须可逆; 不可逆时由 ``bm.linalg.solve`` 抛出当前
            后端对应的线性代数异常. 实现求解 ``K_ii^{-1} K_ib`` 而不显式构造逆矩阵,
            并复用该结果同时给出 ``N`` 与 ``K_s``.

            利用 ``K_local`` 的对称性, ``K_bi`` 由 ``K_ib`` 转置得到而不单独提取.
            这省去一次形状 ``(..., n_b, n_i)`` 的高级索引拷贝, 并消除 ``K_bi`` 与
            ``K_ib`` 之间的不一致来源: 装配得到的 ``K_local`` 若带有舍入级非对称,
            该误差不再传入 ``K_s``, 使下游特征值检查与 Cholesky 分解面对的非对称
            仅来自 ``solve`` 与矩阵乘法本身. 注意这不保证 ``K_s`` 逐位对称. 若传入
            明显非对称的 ``K_local``, 结果等价于对其对称部分做缩聚.

            实现不对 ``K_local`` 做 dtype 转换, 输出 dtype 与输入一致; PyTorch 后端
            下也因此保留计算图, 可直接参与自动微分.
        """
        self._check_local_stiffness(K_local)

        # 按内部和接口自由度索引提取刚度分块; ``...`` 保留全部前导批量维.
        K_ii = K_local[..., self.i_dofs[:, None], self.i_dofs]
        K_ib = K_local[..., self.i_dofs[:, None], self.b_dofs]
        K_bb = K_local[..., self.b_dofs[:, None], self.b_dofs]

        # bm.linalg.solve 沿前导维广播, 对每个子结构独立求解 K_ii^{-1} K_ib.
        invK_ii_K_ib = bm.linalg.solve(K_ii, K_ib)

        # 保存缩聚结果, 供全局接口装配和内部位移恢复复用.
        self.N = -invK_ii_K_ib
        self.K_s = K_bb - bm.matrix_transpose(K_ib) @ invK_ii_K_ib

        return self.K_s, self.N


class StreamingShapeFunctionCondensation(StaticCondensationBase):
    """带同质复用与分块流式特性的形函数缩聚容器 (Streaming & Homogeneous Bypass).

    该类专为大规模/超大规模（如数百万自由度）子结构拓扑优化设计：
    1. 同质子结构（如纯实体/纯孔洞，占全场 80%+）直接按标量因子缩放预计算的基准刚度；
    2. 异质子结构（边界过渡带）采用流式分块求解，避免在内存中一次性分配数十 GB 的局部矩阵；
    3. 支持全局装配与全场内部细观位移的流式批量恢复。
    """

    def __init__(
        self,
        i_dofs: Any,
        b_dofs: Any,
        Ks_batch: Optional[Any] = None,
        N_homo: Optional[Any] = None,
        N_hetero_dict: Optional[Dict[int, Any]] = None,
        is_homo: Optional[Any] = None,
        *,
        Ks_solid: Optional[Any] = None,
        coef_homo: Optional[Any] = None,
        Ks_hetero_dict: Optional[Dict[int, Any]] = None,
        n_sub_total: Optional[int] = None,
    ) -> None:
        super().__init__(i_dofs, b_dofs)
        self.K_s = bm.asarray(Ks_batch) if Ks_batch is not None else None
        self.N_homo = bm.to_numpy(N_homo) if N_homo is not None else None
        self.N_hetero_dict = N_hetero_dict or {}
        self.is_homo = bm.to_numpy(is_homo) if is_homo is not None else None

        # 轻量流式模式属性 (避免全量预分配数十 GB 稠密张量)
        self.Ks_solid = bm.asarray(Ks_solid) if Ks_solid is not None else None
        self.coef_homo = bm.asarray(coef_homo) if coef_homo is not None else None
        self.Ks_hetero_dict = Ks_hetero_dict or {}
        self.n_sub_total = n_sub_total if n_sub_total is not None else (
            len(self.is_homo) if self.is_homo is not None else (
                self.K_s.shape[0] if self.K_s is not None else 0
            )
        )

    def condense(self, K_local: Any = None, rho_local: Optional[Any] = None) -> Tuple[Any, Any]:
        """返回已在流式分块阶段计算完毕的缩聚刚度矩阵 (若未全量存储则返回 None)."""
        return self.K_s, None

    def get_chunk_stiffness(self, start: int, end: int) -> Any:
        """流式获取指定切片范围 [start, end) 的子结构缩聚刚度张量."""
        if self.K_s is not None:
            return self.K_s[start:end]

        n_chunk = end - start
        chunk_Ks = bm.zeros((n_chunk, self.n_b, self.n_b), dtype=bm.float64)
        is_homo_chunk = self.is_homo[start:end]

        # 1. 同质子结构批量广播标量缩放
        homo_rel_idx = np.where(is_homo_chunk)[0]
        if len(homo_rel_idx) > 0 and self.Ks_solid is not None and self.coef_homo is not None:
            coefs = self.coef_homo[start + homo_rel_idx]
            chunk_Ks = bm.set_at(
                chunk_Ks,
                (homo_rel_idx, slice(None), slice(None)),
                coefs[:, None, None] * self.Ks_solid[None, :, :],
            )

        # 2. 异质子结构填充
        hetero_rel_idx = np.where(~is_homo_chunk)[0]
        for r_idx in hetero_rel_idx:
            g_idx = start + r_idx
            if g_idx in self.Ks_hetero_dict:
                chunk_Ks = bm.set_at(
                    chunk_Ks,
                    (r_idx, slice(None), slice(None)),
                    bm.asarray(self.Ks_hetero_dict[g_idx], dtype=bm.float64),
                )

        return chunk_Ks

    def get_projected_stiffness(self, trace_basis: TraceBasis) -> Any:
        """获取指定接口迹空间上的批量子结构刚度.

        参数:
            trace_basis: 从迹自由度到完整接口自由度的线性映射.

        返回:
            全场各子结构的迹空间刚度张量, 形状
            ``(B, n_trace, n_trace)``.
        """
        B = self.n_sub_total
        n_trace = trace_basis.n_trace_dofs
        if self.K_s is not None:
            return trace_basis.project_stiffness(self.K_s)

        Ks_reduced = bm.zeros((B, n_trace, n_trace), dtype=bm.float64)

        # 1. 同质子结构基准降维刚度, 仅需算一次.
        if self.Ks_solid is not None and self.coef_homo is not None and self.is_homo is not None:
            Ks_solid_reduced = trace_basis.project_stiffness(self.Ks_solid)
            homo_idx = bm.nonzero(self.is_homo)[0]
            if len(homo_idx) > 0:
                coefs = self.coef_homo[homo_idx]
                Ks_reduced = bm.set_at(
                    Ks_reduced,
                    (homo_idx, slice(None), slice(None)),
                    coefs[:, None, None] * Ks_solid_reduced[None, :, :],
                )

        # 2. 异质子结构独立降维, 仅极少数边界子结构.
        if len(self.Ks_hetero_dict) > 0:
            for idx, Ks_h in self.Ks_hetero_dict.items():
                Ks_reduced_h = trace_basis.project_stiffness(
                    bm.asarray(Ks_h, dtype=bm.float64)
                )
                Ks_reduced = bm.set_at(Ks_reduced, idx, Ks_reduced_h)

        return Ks_reduced

    def recover_from_trace(
        self,
        trace_displacement: Any,
        trace_basis: TraceBasis,
    ) -> Tuple[Any, Any]:
        """由迹自由度位移恢复完整接口位移与内部细观位移.

        参数:
            trace_displacement: 各子结构迹自由度位移, 形状
                ``(B, n_trace)``.
            trace_basis: 当前接口迹空间.

        返回:
            (u_b_batch, u_i_batch): 细边界位移 (B, n_b) 与内部位移 (B, n_i).
        """
        u_b_batch = trace_basis.expand_displacement(trace_displacement)
        u_i_batch = self.recover(u_b_batch)
        return u_b_batch, u_i_batch

    def get_macro_stiffness(self, L: Any) -> Any:
        """兼容旧接口: 使用角点线性迹计算宏观刚度."""
        return self.get_projected_stiffness(LinearCornerTraceBasis(L))

    def recover_from_macro(self, u_c_batch: Any, L: Any) -> Tuple[Any, Any]:
        """兼容旧接口: 由角点线性迹位移恢复细尺度位移."""
        return self.recover_from_trace(
            u_c_batch,
            LinearCornerTraceBasis(L),
        )

    def recover(self, u_b_batch: Any) -> Any:
        """根据接口位移批量恢复全场子结构内部细观位移."""
        B = u_b_batch.shape[0]
        n_i = self.n_i
        u_i = np.zeros((B, n_i), dtype=np.float64)
        u_b_np = bm.to_numpy(u_b_batch)

        # 1. 同质子结构位移恢复 (单次矩阵乘法批量广播)
        if self.is_homo is not None and self.N_homo is not None:
            homo_idx = np.where(self.is_homo)[0]
            if len(homo_idx) > 0:
                u_i[homo_idx] = u_b_np[homo_idx] @ self.N_homo.T

        # 2. 异质子结构位移恢复
        for idx, N_h in self.N_hetero_dict.items():
            u_i[idx] = bm.to_numpy(N_h) @ u_b_np[idx]

        return bm.asarray(u_i)
