"""子结构静力缩聚.

提供基于 FEALPy ``bm`` 后端的精确 Schur 补缩聚实现. 所有缩聚接口都接受任意
可变前导维 ``...``, 单个子结构只是前导维为空的特例, 批量子结构对应前导维 ``B``.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from fealpy.backend import backend_manager as bm


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
