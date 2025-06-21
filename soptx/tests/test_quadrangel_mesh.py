"""测试 QuadrangleMesh 中 from_box 的功能"""
from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh

bm.set_backend('pytorch')
mesh = QuadrangleMesh.from_box(
                    box=[0, 1, 0, 1], 
                    nx=10, ny=10,
                    device='cuda'
                )
print("---------------")
