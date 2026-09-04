"""接口系统的 Matrix-Free 算子.

``GlobalAssembler.assemble_interface_system`` 把各子结构的缩聚刚度散加成显式
``CSRTensor``; 本模块提供同一装配的算子形式: 保留 ``(B, n_b, n_b)`` 的批量缩聚
刚度, 每次作用时按接口编号 gather, 做批量 GEMV, 再散加回接口向量, 全程不形成
全局接口矩阵.

两者代数恒等, 不是近似. 显式装配就是把同一批稠密块按同一批索引求和, 因此

    A x = scatter-add_b ( K_s_batch @ x[b] )

与 ``system.stiffness @ x`` 的差异只来自浮点求和次序. 一致性判据据此取
``1e-13`` 量级: 超出该量级的偏差指向索引错误, 而不是精度损失.

算子直接定义在自由子空间上, 与 ``solve_interface_system`` 中
``stiffness[free, free]`` 的语义一致: 受约束自由度既不进入输入也不进入输出.
这样算子谱中不含被约束方向引入的伪零模, 可直接用于条件数与特征值分析.

``apply_full`` 另外保留全接口空间上的作用入口: 非齐次 Dirichlet 的右端修正需要
反力项 ``-(K u_c)_free``, 其中 ``u_c`` 支撑在固定自由度上, 该作用无法由自由子
空间上的 ``__matmul__`` 表达; 复用同一实例即可, 不必为此再构造一个无约束算子.
"""

from typing import Any, Optional, Sequence

from fealpy.backend import backend_manager as bm

from .assembler import GlobalAssembler


class InterfaceOperator:
    """接口刚度矩阵的 Matrix-Free 作用, 定义在自由子空间上.

    属性:
        global_dofs: 接口自由度对应的全局自由度编号, 升序, 形状 ``(n_interface,)``.
        indices: 各子结构接口自由度在接口向量中的编号, 形状 ``(B, n_b)``. 即算子
            代数中的 gather 映射 ``G_j``.
        K_s_batch: 批量缩聚刚度, 形状 ``(B, n_b, n_b)``. 来源不限: 精确 Schur 补,
            PIML 预测或人工构造的对称矩阵, 算子对此无感.
        free: 自由接口自由度的编号, 形状 ``(n_free,)``.
        shape: ``(n_free, n_free)``, 供 Krylov 求解器查询.
    """

    def __init__(
        self,
        assembler: GlobalAssembler,
        sub_meshes: Sequence[Any],
        condensors: Any,
        *,
        fixed_dofs: Optional[Any] = None,
    ) -> None:
        """构造算子并缓存映射与批量刚度.

        参数:
            assembler: 已构造的全局装配器.
            sub_meshes: 子结构列表.
            condensors: 缩聚器列表或单个批量缩聚器, 形状约定与
                ``GlobalAssembler.assemble_interface_system`` 相同.
            fixed_dofs: 受约束的**接口**自由度编号, 可由
                ``GlobalAssembler.project_global_dofs`` 从全局固定自由度投影得到.
                为 ``None`` 时算子定义在全部接口自由度上.

        说明:
            映射与批量刚度在构造时算一次并缓存; 每次作用只做 gather, 批量 GEMV
            与散加, 不重复查询自由度.
        """
        if not sub_meshes:
            raise ValueError("sub_meshes 不能为空.")

        self.global_dofs = assembler.build_interface_dofs(sub_meshes)
        self.n_interface = int(len(self.global_dofs))

        n_b = int(sub_meshes[0].n_b)
        self.indices = assembler.interface_indices(sub_meshes, self.global_dofs)
        self.K_s_batch, _ = assembler.normalize_condensors(
            condensors, len(sub_meshes), n_b
        )
        self._flat_indices = bm.reshape(self.indices, (-1,))

        all_dofs = bm.arange(self.n_interface, dtype=bm.int64)
        if fixed_dofs is None:
            self.fixed = bm.zeros((0,), dtype=bm.int64)
            self.free = all_dofs
        else:
            self.fixed = bm.asarray(fixed_dofs, dtype=bm.int64)
            self.free = all_dofs[bm.isin(all_dofs, self.fixed, invert=True)]
        self.n_free = int(len(self.free))
        self.shape = (self.n_free, self.n_free)

    def __matmul__(self, x_free: Any) -> Any:
        """算子作用: gather, 批量 GEMV, 散加, 再取回自由子空间.

        参数:
            x_free: 自由接口自由度上的向量, 形状 ``(n_free,)``.

        返回:
            y_free: 形状 ``(n_free,)``, 与 ``stiffness[free, free] @ x_free`` 相等.

        说明:
            散加用 ``bm.index_add`` 而不是 ``bm.add_at``: 接口自由度被相邻子结构
            共享, ``indices`` 必然含重复项, 而 PyTorch 后端的 ``add_at`` 实现为
            ``a[indices] += src``, 重复索引下只保留任意一项, 结果错误且不报错.
        """
        if int(len(x_free)) != self.n_free:
            raise ValueError(
                f"x_free 的长度必须等于自由接口自由度数 {self.n_free}; "
                f"当前为 {int(len(x_free))}."
            )
        context = bm.context(self.K_s_batch)

        x = bm.zeros((self.n_interface,), **context)
        x = bm.set_at(x, self.free, x_free)
        return self.apply_full(x)[self.free]

    def apply_full(self, x: Any) -> Any:
        """全接口空间上的作用 ``y = sum_j L_j^T K_s^j L_j x``, 不区分自由/固定.

        参数:
            x: 全部接口自由度上的向量, 形状 ``(n_interface,)``.

        返回:
            y: 形状 ``(n_interface,)``, 与 ``system.stiffness @ x`` 相等.

        说明:
            这是算子的核心三步 (gather, 批量 GEMV, 散加); ``__matmul__`` 等于
            "嵌入自由子空间 -> ``apply_full`` -> 取回自由分量". 非齐次 Dirichlet
            的反力项取 ``apply_full(u_c)[free]``, 其中 ``u_c`` 只在固定自由度上
            非零.
        """
        if int(len(x)) != self.n_interface:
            raise ValueError(
                f"x 的长度必须等于接口自由度数 {self.n_interface}; "
                f"当前为 {int(len(x))}."
            )
        x_b = x[self.indices]
        y_b = bm.einsum('bij, bj -> bi', self.K_s_batch, x_b)

        y = bm.zeros((self.n_interface,), **bm.context(self.K_s_batch))
        return bm.index_add(y, self._flat_indices, bm.reshape(y_b, (-1,)))

    def diagonal(self) -> Any:
        """自由子空间上的算子对角, 形状 ``(n_free,)``, 供 Jacobi 类预条件使用.

        说明:
            ``diag(A) = sum_j L_j^T diag(K_s^j)``: 逐子结构取对角后按同一
            ``_flat_indices`` 散加, 与 ``apply_full`` 共享 gather/scatter 语义,
            代价为一次批量取对角与一次散加, 不做任何算子作用.
        """
        diag_b = bm.einsum('bii -> bi', self.K_s_batch)

        d = bm.zeros((self.n_interface,), **bm.context(self.K_s_batch))
        d = bm.index_add(d, self._flat_indices, bm.reshape(diag_b, (-1,)))
        return d[self.free]

    def matvec(self, x_free: Any) -> Any:
        """``__matmul__`` 的别名, 供只认 ``matvec`` 的求解器调用."""
        return self @ x_free

    def to_dense(self) -> Any:
        """把算子在自由子空间上展开成稠密矩阵, 形状 ``(n_free, n_free)``.

        说明:
            逐列作用于单位向量, 代价为 ``n_free`` 次算子作用. 只用于接口自由度数
            很小时的特征值与条件数分析, 不用于求解路径.
        """
        context = bm.context(self.K_s_batch)
        columns = []
        for k in range(self.n_free):
            unit = bm.zeros((self.n_free,), **context)
            unit = bm.set_at(unit, k, 1.0)
            columns.append(self @ unit)
        return bm.stack(columns, axis=1)
