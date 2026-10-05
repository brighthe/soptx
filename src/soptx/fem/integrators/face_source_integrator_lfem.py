"""Lagrange 有限元的面源项积分子.

``_FaceSourceIntegrator`` 在选定的面上计算局部载荷向量 ``(g, v)_F``, 子类只需实现
``make_index`` 给出参与积分的面编号. ``LagrangeBoundarySourceIntegrator`` 用于边界面上
的自然边界 (牵引) 载荷.
"""

from typing import Optional

from soptx.backend import backend_manager as bm
from soptx.typing import TensorLike, SourceLike, Threshold
from soptx.mesh import HomogeneousMesh
from soptx.functionspace.space import FunctionSpace as _FS
from soptx.fem.integrator import LinearInt, SrcInt, FaceInt, enable_cache


class _FaceSourceIntegrator(LinearInt, SrcInt, FaceInt):
    def __init__(self, 
                source: Optional[SourceLike] = None, 
                q: Optional[int] = None, *,
                threshold: Optional[Threshold] = None,
                batched: bool = False):
        super().__init__()
        self.source = source 
        self.q = q
        self.threshold = threshold
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space) -> TensorLike:
        """选定面到全局自由度的映射.

        Parameters
        ----------
        space : FunctionSpace
            检验函数空间.

        Returns
        -------
        TensorLike
            ``space.face_to_dof(index=index)``, 形状 ``(NF_sel, ldof_face)``, 其中
            ``index`` 由 ``make_index`` 给出.
        """
        index = self.make_index(space)
        return space.face_to_dof(index=index)

    @enable_cache
    def fetch(self, space: _FS) -> TensorLike:
        """取选定面上的求积数据与面基函数, 结果按空间缓存.

        Parameters
        ----------
        space : FunctionSpace
            检验函数空间, 须定义在齐次网格上.

        Returns
        -------
        bcs : TensorLike
            面积分点的重心坐标.
        ws : TensorLike
            积分权重, 形状 ``(NQ_face, )``.
        phi : TensorLike
            面基函数值; 对向量值张量空间形状为 ``(1, NQ_face, ldof_face, GD)``.
        facemeasure : TensorLike
            选定面的测度, 形状 ``(NF_sel, )``.
        index : TensorLike
            ``make_index`` 给出的面编号.

        Raises
        ------
        RuntimeError
            网格不是 ``HomogeneousMesh``.

        Notes
        -----
        积分阶默认取 ``space.p + 3``.
        """
        index = self.make_index(space)
        mesh = space.mesh

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomogeneousMesh.")

        facemeasure = mesh.entity_measure('face', index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.face_basis(bcs, index=index) # (1, NQ_face, LDOF_face, GD)

        return bcs, ws, phi, facemeasure, index
    
    def assembly(self, space):
        """计算选定面上的局部载荷向量 ``(g, v)_F``.

        Parameters
        ----------
        space : FunctionSpace
            向量值检验函数空间 (面基函数带末尾分量轴, 如 ``TensorFunctionSpace``).

        Returns
        -------
        TensorLike
            形状 ``(NF_sel, ldof_face)`` 的局部载荷向量.

        Notes
        -----
        ``source`` 须为可调用对象, 在面积分点的物理坐标上求值, 返回形状
        ``(NF_sel, NQ_face, GD)``. 本方法不读 ``batched``.
        """
        source = self.source
        bcs, ws, phi, fm, index = self.fetch(space) 
        mesh = getattr(space, 'mesh', None)

        ps = mesh.bc_to_point(bcs, index=index)
        val = source(ps) # (NF_bd, NQ_face, GD)

        F = bm.einsum('f, q, qld, fqd -> fl', fm, ws, phi[0], val) # (NF_bd, LDOF_face)

        return F

class InterFaceSourceIntegrator(_FaceSourceIntegrator):
    """以 ``threshold`` 直接作为面编号的面源项积分子."""

    def make_index(self, space: _FS):
        """返回参与积分的面编号.

        Parameters
        ----------
        space : FunctionSpace
            检验函数空间 (未使用).

        Returns
        -------
        TensorLike
            ``threshold`` 原样返回, 应为面编号.
        """
        index = self.threshold

        return index

class LagrangeBoundarySourceIntegrator(_FaceSourceIntegrator):
    """边界面上的自然边界载荷积分器.

    Note
    ----
    ``threshold`` 为 callable 时, 判定只在**面重心**上做一次, 因此选面是整面
    全有或全无的: 重心落在载荷区内, 整个面按 ``source`` 积分; 落在区外, 整个面
    完全不参与. 载荷区边界落在某个面内部时, 该面会被整体计入或整体丢弃, 离散
    合力随之相对解析合力偏大或偏小最多一个面的贡献, 且没有任何报错.

    因此载荷区端点应与网格面的端点对齐 (本仓库全部基准算例如此); 做不到对齐时,
    应改用 ``soptx.fem.boundary_loads.project_patch_traction_to_p1_trace`` 把
    局部牵引先投影到 P1 迹空间 —— 投影精确保持合力与一阶矩, 与网格是否对齐无关.
    装配后可用 ``soptx.fem.boundary_loads.check_boundary_load_resultant``
    核对离散合力.
    """

    def make_index(self, space: _FS):
        """按 ``threshold`` 选出参与积分的边界面编号.

        Parameters
        ----------
        space : FunctionSpace
            检验函数空间.

        Returns
        -------
        TensorLike
            面编号. ``threshold`` 为张量时原样返回; 为 None 时取全部边界面; 为单个
            callable 时取重心使其为 True 的边界面; 为 tuple 或 list 时取满足其中任一
            callable 的边界面, 非 callable 元素被忽略, 全部不是 callable 时取全部边界面.
        """
        threshold = self.threshold

        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            mesh = space.mesh
            index = mesh.boundary_face_index()
            if isinstance(threshold, (tuple, list)):
                # threshold 是元组或列表, 包含多个边界判断函数
                bc = mesh.entity_barycenter('face', index=index)
                flags = []
                for thresh_func in threshold:
                    if callable(thresh_func):
                        flags.append(thresh_func(bc))
                if flags:
                    combined_flag = flags[0]
                    for flag in flags[1:]:
                        combined_flag = combined_flag | flag
                    index = index[combined_flag]
            elif callable(threshold):
                # threshold 是单个 callable 函数
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]


        return index
