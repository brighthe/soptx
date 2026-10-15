"""胡张 (Hu-Zhang) 对称应力有限元空间的工厂, 按 ``mesh.top_dimension()`` 分派到 2D/3D 实现."""

from soptx.mesh import MeshView

class HuZhangFESpace:
    """Factory class for creating HuZhang finite element spaces."""
    def __new__(cls, mesh: MeshView, p: int = 1, ctype: str = 'C', use_relaxation: bool = False,
                corners=None, traction_face=None, traction_edge=None, symmetry_edge=None):
        TD = mesh.top_dimension()

        if TD == 2:
            if traction_face is not None:
                raise ValueError("traction_face 只用于三维胡张元的标架对齐, 二维不接受.")
            from .huzhang_fe_space_2d import HuZhangFESpace2d
            return HuZhangFESpace2d(mesh, p=p, ctype=ctype, use_relaxation=use_relaxation,
                                    corners=corners, traction_edge=traction_edge,
                                    symmetry_edge=symmetry_edge)
        elif TD == 3:
            if use_relaxation:
                raise ValueError("三维胡张元没有角点松弛, use_relaxation 只能为 False.")
            if traction_edge is not None or symmetry_edge is not None:
                raise ValueError("traction_edge / symmetry_edge 只用于二维胡张元, 三维请用 traction_face.")
            from .huzhang_fe_space_3d import HuZhangFESpace3d
            return HuZhangFESpace3d(mesh, p=p, ctype=ctype, traction_face=traction_face)
        else:
            raise ValueError(f"Unsupported dimension: {TD}. Only 2D and 3D are supported.")
