from fealpy.backend import backend_manager as bm

bm.set_backend('pytorch')

def test_2d_filter_matrix(device):
    '''
    准确: 四边形网格, 三者的结果一致
    效率: CPU 下, _compute_filter_2d 的速度最快,
                 _compute_filter_2d_math 的速度次之, 
                 _compute_filter_general 最慢
    '''
    from soptx.filter.matrix_builder import FilterMatrixBuilder

    from soptx.pde import HalfMBBBeam2dData1

    pde = HalfMBBBeam2dData1(domain=[0, 1, 0, 1])
    mesh = pde.create_mesh(mesh_type='quadrangle', nx=60, ny=20, threshold=None, device=device)


    rmin = mesh.meshdata['nx'] * 0.04
    builder = FilterMatrixBuilder(mesh=mesh, rmin=rmin)
    # ? 不能同时调用两个, 会报错
    H1, Hs1 = builder._compute_filter_2d(
                                rmin=rmin,
                                nx=mesh.meshdata['nx'], ny=mesh.meshdata['ny'],
                                hx=mesh.meshdata['hx'], hy=mesh.meshdata['hy'],
                                )
    cell_centers = mesh.entity_barycenter('cell')


    # H2, Hs2 = builder._compute_filter_2d_math(
    #                             rmin=rmin,
    #                             nx=mesh.meshdata['nx'], ny=mesh.meshdata['ny'],
    #                             hx=mesh.meshdata['hx'], hy=mesh.meshdata['hy'],
    #                             )



    H3, Hs3 = builder._compute_filter_general(
                                rmin=rmin,
                                domain=mesh.meshdata['domain'],
                                cell_centers=cell_centers
                            )

    # error = bm.sum(bm.abs(H1.toarray() - H2.toarray()))
    error1 = bm.sum(bm.abs(H1.toarray() - H3.toarray()))
    print("----------------")

if __name__ == "__main__":

    test_2d_filter_matrix(device='cuda')
    print("FilterMatrixBuilder test completed successfully.")
    
    
