from fealpy.backend import backend_manager as bm

bm.set_backend('pytorch')

def test_2d_filter_matrix(mesh):
    cell_centers = mesh.entity_barycenter('cell')
    print("-------------------------------")

if __name__ == "__main__":

    from soptx.pde import HalfMBBBeam2dData1

    pde = HalfMBBBeam2dData1(domain=[0, 1, 0, 1])
    mesh = pde.create_mesh(mesh_type='quadrangle', nx=60, ny=20, device='cuda')

    test_2d_filter_matrix(mesh)
    print("FilterMatrixBuilder test completed successfully.")