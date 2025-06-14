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
                        origin=[0.0, 0.0])
    cip1 = mesh.cell_to_ipoint(p=p)
    print(f"cip(nx != ny):\n{cip1}")

    nx, ny = 3, 3
    hx, hy = 1, 1
    mesh2 = UniformMesh2d(extent=[0, nx, 0, ny], 
                        h=[hx, hy], 
                        origin=[0.0, 0.0])
    cip2 = mesh2.cell_to_ipoint(p=p)
    print(f"cip(nx == ny):\n{cip2}")

    # ! 不要使用 UniformMesh
    print(f"在 nx != ny 的情况下, cell_to_ipoint 的结果是错误的!")

if __name__ == "__main__":
    test_uniform_mesh_2d_cell_to_ipoint()
