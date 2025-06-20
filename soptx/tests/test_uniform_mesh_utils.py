from fealpy.backend import backend_manager as bm

def test_uniform_mesh_2d_cell_to_ipoint():
    """
    测试 UniformMesh2d.cell_to_ipoint() 方法的正确性,
        包括 nx == ny 和 nx != ny 的情况
    """
    from fealpy.mesh import UniformMesh2d

    p = 2

    nx, ny = 4, 3
    hx, hy = 1, 1
    mesh = UniformMesh2d(extent=[0, nx, 0, ny], 
                        h=[hx, hy], 
                        origin=[0.0, 0.0],
                        ipoints_ordering='nec')
    cip1 = mesh.cell_to_ipoint(p=p)
    print(f"cip(nx != ny):\n{cip1}")

    nx, ny = 3, 3
    hx, hy = 1, 1
    mesh2 = UniformMesh2d(extent=[0, nx, 0, ny], 
                        h=[hx, hy], 
                        origin=[0.0, 0.0],
                        ipoints_ordering='nec')
    cip2 = mesh2.cell_to_ipoint(p=p)
    print(f"cip(nx == ny):\n{cip2}")

    # ! 不要使用 UniformMesh
    print(f"在 nx != ny 的情况下, cell_to_ipoint 的结果是错误的!")

def test_quadrangle_mesh_2d_cell_to_ipoint():
    """
    测试 QuadrangleMesh.cell_to_ipoint() 方法的正确性,
        包括 nx == ny 和 nx != ny 的情况
    """
    from fealpy.mesh import QuadrangleMesh

    nx, ny = 4, 3
    hx, hy = 1, 1
    mesh = QuadrangleMesh.from_box(box=[0, nx*hx, 0, ny*hy], nx=nx, ny=ny)

    cip_1 = mesh.cell_to_ipoint(p=1)
    print(f"cip(p=1):\n{cip_1}")
    cip_2 = mesh.cell_to_ipoint(p=2)
    print(f"cip(p=2):\n{cip_2}")

if __name__ == "__main__":
    test_uniform_mesh_2d_cell_to_ipoint()
    print("--------------------------")
    test_quadrangle_mesh_2d_cell_to_ipoint()
