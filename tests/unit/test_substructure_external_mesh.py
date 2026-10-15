"""外部规则网格复用、编号一致性与不匹配输入检查."""

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.fem.substructure import StructuredSubstructureLayout, build_substructures
from soptx.mesh import HexahedronMesh, QuadrangleMesh


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('degree', [1, 2])
def test_external_mesh_reused_with_identical_numbering(dim, degree):
    """复用同一对象, 且位移空间及接口编号与惰性构造一致."""
    bm.set_backend('numpy')
    size = (4.0, 6.0, 8.0)[:dim]
    grid = (4,) * dim
    box = [v for length in size for v in (0.0, length)]
    factory = QuadrangleMesh if dim == 2 else HexahedronMesh
    mesh = factory.from_box(box, *grid)
    options = dict(domain_size=size, n_sub=(2,) * dim, n_fine=(2,) * dim, degree=degree)
    layout = StructuredSubstructureLayout(**options, full_mesh=mesh)
    reference = StructuredSubstructureLayout(**options)
    assert layout.full_mesh is mesh
    np.testing.assert_array_equal(layout.space_full.cell_to_dof(), reference.space_full.cell_to_dof())
    _, sub_meshes, _ = build_substructures(layout)
    np.testing.assert_array_equal(
        layout.build_interface_dofs(sub_meshes), reference.build_interface_dofs(sub_meshes),
    )


@pytest.mark.parametrize('mismatch', ['size', 'grid', 'nodes', 'cells'])
def test_external_mesh_mismatch_rejected(mismatch):
    """拒绝尺寸、剖分、节点或单元编号不匹配的网格."""
    bm.set_backend('numpy')
    box = [0.0, 5.0 if mismatch == 'size' else 4.0, 0.0, 4.0, 0.0, 4.0]
    mesh = HexahedronMesh.from_box(box, 3 if mismatch == 'grid' else 4, 4, 4)
    if mismatch == 'nodes':
        mesh.entity('node')[[0, 1]] = mesh.entity('node')[[1, 0]]
    if mismatch == 'cells':
        mesh.entity('cell')[[0, 1]] = mesh.entity('cell')[[1, 0]]
    with pytest.raises(ValueError, match='full_mesh'):
        StructuredSubstructureLayout(
            domain_size=(4.0,) * 3, n_sub=(2,) * 3, n_fine=(2,) * 3, full_mesh=mesh,
        )
