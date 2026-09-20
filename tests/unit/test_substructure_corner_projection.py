"""全局角点投影的仿射位移再现测试."""
from types import SimpleNamespace

import numpy as np
from fealpy.backend import backend_manager as bm
from soptx.fem.substructure import GlobalAssembler, LinearCornerTraceBasis, build_substructures


def test_shared_interface_reproduces_affine_displacement():
    """跨两个子结构的全局插值应再现同一个仿射位移场."""
    bm.set_backend("numpy")
    assembler = GlobalAssembler((2., 1.), (2, 1), (2, 2))
    prototype, meshes, _ = build_substructures(assembler)
    interface = assembler.build_interface_dofs(meshes)
    projection = assembler.build_linear_corner_projection(
        meshes, SimpleNamespace(global_dofs=interface),
        LinearCornerTraceBasis.from_prototype(prototype),
    )
    def affine(nodes):
        x, y = np.asarray(bm.to_numpy(nodes)).T
        return np.column_stack((1 + 2*x - y, -2 + x + 3*y)).reshape(-1)
    macro = affine(assembler.macro_node_coordinates())
    expected = affine(assembler.full_mesh.entity("node"))[bm.to_numpy(interface)]
    np.testing.assert_allclose(projection @ macro, expected, atol=1e-12)
    assert projection.shape == (len(interface), assembler.total_macro_dofs)
