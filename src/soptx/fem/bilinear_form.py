# 部分移植自 brighthe/fealpy ``fealpy/fem/bilinear_form.py`` @ f474a5775 (coalesce 装配路线、
# 形状检查与 ``@``), 与 SOPTX 原子类合并为单个类.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""SOPTX 有限元双线性型 (BilinearForm) 模块.

``BilinearForm`` 是全装配 (FA) 层级的双线性型门面, 积分子容器与 ``add_integrator`` /
``__lshift__`` 等接口来自 ``soptx.fem.form.Form``. ``assembly()`` 提供两条装配路线:

pattern (默认)
    先按 ``spaces[0]`` 惰性建好 ``CSRPattern`` 骨架与槽位映射并缓存, 再把各组单元刚度
    相加后用 ``bm.add_at`` 原地累加 (CPU 上 ``np.add.at``, GPU 上 ``scatter_add_``),
    不经过 COO 排序. 只适用于单空间、单列、全部积分子按 ``cell_to_dof`` 给出局部张量
    的情形.
coalesce
    逐组把局部张量展开成 ``COOTensor`` 后相加, 再 ``coalesce`` 并转成 CSR. 适用于
    双空间 (矩形) 矩阵、面积分子与批量装配, 是 pattern 路线的通用后备.

EA 及以下的无矩阵算子在 ``soptx.fem.levels`` 中, 不经过本类的 ``assembly()``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Union, overload

from soptx.backend import backend_manager as bm
from soptx.fem.form import Form
from soptx.fem.integrator import LinearInt
from soptx.fem.matrix.csr_pattern import CSRPattern, assemble_csr, build_csr_pattern
from soptx.sparse import COOTensor, CSRTensor
from soptx.typing import TensorLike

logger = logging.getLogger(__name__)


class BilinearForm(Form[LinearInt]):
    """有限元双线性型.

    Parameters
    ----------
    space : FunctionSpace or tuple of FunctionSpace
        单个空间时试验与检验空间相同; 二元组 ``(trial, test)`` 时装配
        ``(test_gdof, trial_gdof)`` 的矩形矩阵.
    *args
        原样传给 ``Form``.
    pattern : CSRPattern, optional
        预建的 CSR 骨架. 为 None 时由 pattern 路线首次装配时建好并缓存.
    **kwargs
        原样传给 ``Form``, 如 ``batch_size``.
    """

    _M = None

    def __init__(
        self,
        space: Any,
        *args: Any,
        pattern: Optional[CSRPattern] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(space, *args, **kwargs)
        self._pattern: Optional[CSRPattern] = pattern

    @property
    def pattern(self) -> Optional[CSRPattern]:
        """当前绑定的 CSR 骨架."""
        return self._pattern

    @pattern.setter
    def pattern(self, value: Optional[CSRPattern]) -> None:
        """设置或更新 CSR 骨架."""
        self._pattern = value

    def _get_sparse_shape(self):
        """返回全局矩阵形状 ``(test_gdof, trial_gdof)``."""
        spaces = self._spaces
        ugdof = spaces[0].number_of_global_dofs()
        vgdof = spaces[1].number_of_global_dofs() if (len(spaces) > 1) else ugdof
        return (vgdof, ugdof)

    def check_local_shape(self, entity_to_global: TensorLike, local_tensor: TensorLike):
        """检查积分子给出的局部张量与实体到全局自由度映射是否匹配.

        Parameters
        ----------
        entity_to_global : TensorLike
            ``(NE, ldof)`` 的实体到全局自由度映射.
        local_tensor : TensorLike
            ``(NE, vldof, uldof)`` 或带批量维的 ``(B, NE, vldof, uldof)`` 局部张量.

        Raises
        ------
        ValueError
            维数或实体数不匹配.
        """
        if entity_to_global.ndim != 2:
            raise ValueError("entity-to-global relationship should be a 2D tensor, "
                             f"but got shape {tuple(entity_to_global.shape)}.")
        if entity_to_global.shape[0] != local_tensor.shape[0]:
            raise ValueError(f"entity_to_global.shape[0] != local_tensor.shape[0]")
        if local_tensor.ndim not in (3, 4):
            raise ValueError("Output of operator integrators should be 3D "
                             "(or 4D with batch in the first dimension), "
                             f"but got shape {tuple(local_tensor.shape)}.")

    def check_space(self):
        """检查空间个数, 以及双空间时两者的设备与浮点类型是否一致.

        Raises
        ------
        ValueError
            空间个数不是 1 或 2, 或两个空间的设备或浮点类型不同.
        """
        if len(self._spaces) not in {1, 2}:
            raise ValueError("BilinearForm should have 1 or 2 spaces, "
                             f"but got {len(self._spaces)} spaces.")
        if len(self._spaces) == 2:
            s0, s1 = self._spaces
            if s0.mesh.device != s1.mesh.device:
                raise ValueError("Spaces should have the same device, "
                                f"but got {s0.device} and {s1.device}.")
            if s0.ftype != s1.ftype:
                raise ValueError("Spaces should have the same dtype, "
                                f"but got {s0.ftype} and {s1.ftype}.")

    @overload
    def assembly(self) -> CSRTensor: ...
    @overload
    def assembly(self, *, format: Literal["csr"], method: str = "pattern") -> CSRTensor: ...
    @overload
    def assembly(self, *, format: Literal["coo"], method: str = "pattern") -> COOTensor: ...
    def assembly(
        self,
        *,
        format: Literal["csr", "coo"] = "csr",
        method: str = "pattern",
    ) -> Union[CSRTensor, COOTensor]:
        """装配全局矩阵.

        Parameters
        ----------
        format : {'csr', 'coo'}, optional
            产出格式, 默认 'csr'.
        method : {'pattern', 'coalesce'}, optional
            装配路线, 默认 'pattern'. 双空间、批量装配或面积分子须用 'coalesce'.

        Returns
        -------
        CSRTensor or COOTensor
            ``([batch, ]test_gdof, trial_gdof)`` 的全局稀疏矩阵.

        Raises
        ------
        ValueError
            ``method`` 或 ``format`` 取值不在支持范围内, 或 pattern 路线的前提不成立.
        RuntimeError
            pattern 路线下未添加任何积分子.
        """
        if method == "coalesce":
            return self._coalesce_assembly(format)

        if method != "pattern":
            raise ValueError(f"不支持的装配路线: {method!r}, 只能是 'pattern' 或 'coalesce'")

        if format not in ("csr", "coo"):
            raise ValueError(f"不支持的产出格式: {format!r}, 只能是 'csr' 或 'coo'")

        K_csr = self._pattern_assembly()

        if getattr(self, "_transposed", False):
            K_csr = K_csr.T

        self._M = K_csr

        if format == "coo":
            return K_csr.tocoo()
        return K_csr

    def _pattern_assembly(self) -> CSRTensor:
        """pattern 路线: 复用 CSR 骨架, 把各组单元刚度相加后原地累加.

        Returns
        -------
        CSRTensor
            ``(gdof, gdof)`` 的全局矩阵.

        Raises
        ------
        ValueError
            双空间、批量装配, 或某组局部张量的形状与骨架不符.
        RuntimeError
            未添加任何积分子.

        Notes
        -----
        骨架只由 ``spaces[0]`` 的单元自由度决定, 积分子给出的 ``entity_to_global`` 不参与
        散加. 因此这里逐组核对局部张量形状是否为 ``(NC, ldof, ldof)``: 面积分子或只在部分
        单元上积分的积分子形状对不上, 会被拒绝. 形状相符但自由度编号与 ``cell_to_dof``
        不同的积分子无法在此识别.
        """
        if len(self._spaces) != 1:
            raise ValueError("pattern 路线只支持单空间; 双空间 (矩形) 矩阵请用 method='coalesce'")
        if self.batch_size != 0:
            raise ValueError("pattern 路线不支持批量装配; 请用 method='coalesce'")

        if self._pattern is None:
            self._pattern = build_csr_pattern(self._spaces[0])

        expected = (self._pattern.n_cells, ) + (self._pattern.scalar_ldof * self._pattern.dof_numel, ) * 2

        K_e = None
        for group_tensor, _ in self.assembly_local_iterative():
            if tuple(group_tensor.shape) != expected:
                raise ValueError(
                    f"局部张量形状 {tuple(group_tensor.shape)} 与 CSR 骨架要求的 {expected} 不符; "
                    "面积分子或部分单元上的积分子请用 method='coalesce'"
                )
            K_e = group_tensor if K_e is None else K_e + group_tensor

        if K_e is None:
            raise RuntimeError("BilinearForm 中未添加任何有效的积分子 (Integrator)")

        return assemble_csr(K_e, self._pattern)

    def _coalesce_assembly(self, format: str) -> Union[CSRTensor, COOTensor]:
        """coalesce 路线: 逐组展开成 COO 后相加, 合并重复项.

        Parameters
        ----------
        format : {'csr', 'coo'}
            产出格式.

        Returns
        -------
        CSRTensor or COOTensor
            ``([batch, ]test_gdof, trial_gdof)`` 的全局稀疏矩阵.

        Raises
        ------
        ValueError
            ``format`` 取值不在支持范围内.
        """
        M = self._scalar_assembly()
        if getattr(self, '_transposed', False):
            M = M.T

        if format == 'csr':
            self._M = M.coalesce().tocsr()
        elif format == 'coo':
            self._M = M.coalesce()
        else:
            raise ValueError(f"Unsupported format {format}.")
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M

    def _scalar_assembly(self) -> COOTensor:
        """把全部积分子的局部张量展开成一个未合并的 ``COOTensor``."""
        self.check_space()
        space = self._spaces
        batch_size = self.batch_size
        ugdof = space[0].number_of_global_dofs()
        vgdof = space[1].number_of_global_dofs() if (len(space) > 1) else ugdof
        init_value_shape = (0,) if (batch_size == 0) else (batch_size, 0)
        sparse_shape = (vgdof, ugdof)

        M = COOTensor(
            indices = bm.empty((2, 0), dtype=space[0].itype, device=bm.get_device(space[0])),
            values = bm.empty(init_value_shape, dtype=space[0].ftype, device=bm.get_device(space[0])),
            spshape = sparse_shape
        )
        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            ue2dof = e2dofs_tuple[0]
            ve2dof = e2dofs_tuple[1] if (len(e2dofs_tuple) > 1) else ue2dof
            local_shape = group_tensor.shape[-3:] # (NC, vldof, uldof)

            if (batch_size > 0) and (group_tensor.ndim == 3): # 积分子不带批量维时沿批量广播
                group_tensor = bm.stack([group_tensor]*batch_size, axis=0)
            I = bm.broadcast_to(ve2dof[:, :, None], local_shape)
            J = bm.broadcast_to(ue2dof[:, None, :], local_shape)
            indices = bm.stack([I.ravel(), J.ravel()], axis=0)
            group_tensor = bm.reshape(group_tensor, self._values_ravel_shape)
            M = M.add(COOTensor(indices, group_tensor, sparse_shape))

        return M

    @property
    def T(self):
        """转置的双线性型: 共享已装配矩阵, 再次装配时产出转置."""
        transposed = self.copy()
        transposed._transposed = True
        transposed._M = self._M
        return transposed

    def __matmul__(self, u: TensorLike):
        """矩阵向量乘 ``A @ u``.

        已装配时直接用全局矩阵; 否则逐组 gather ``u[e2dof]``, 与局部张量做 einsum 后
        ``index_add`` 回全局, 不形成全局矩阵.

        Parameters
        ----------
        u : TensorLike
            ``(trial_gdof, )`` 或 ``(B, trial_gdof)`` 的向量.

        Returns
        -------
        TensorLike
            ``(test_gdof, )`` 或 ``(B, test_gdof)`` 的乘积.
        """
        if self._M is not None:
            return self._M @ u

        nrow = self.shape[-2]
        kwargs = bm.context(u)

        if self.batch_size > 0:
            shape = (self.batch_size, nrow)
            out_subs = 'bci'
            gv_reshape = (self.batch_size, -1)
        else:
            if u.ndim >= 2:
                shape = (u.shape[0], nrow)
                out_subs = 'bci'
                gv_reshape = (u.shape[0], -1)
            else:
                shape = (nrow,)
                out_subs = 'ci'
                gv_reshape = (-1,)

        v = bm.zeros(shape, **kwargs)
        gt_subs = 'bcij' if (self.batch_size > 0) else 'cij'
        gu_subs = 'bcj' if (u.ndim >= 2) else 'cj'

        for group_tensor, e2dofs_tuple in self.assembly_local_iterative():
            ue2dof = e2dofs_tuple[0]
            ve2dof = e2dofs_tuple[1] if (len(e2dofs_tuple) > 1) else ue2dof
            gu = u[..., ue2dof] # (..., NC, uldof)
            gv = bm.einsum(f'{gt_subs}, {gu_subs} -> {out_subs}', group_tensor, gu)
            v = bm.index_add(v, ve2dof.reshape(-1), gv.reshape(gv_reshape))

        return v
