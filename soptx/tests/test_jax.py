from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d

extent = [0, 10, 0, 10]
h = [1.0, 1.0]
origin = [0.0, 0.0]
mesh = UniformMesh2d(
            extent=extent, h=h, origin=origin,
            ipoints_ordering='yx', flip_direction=None,
            device='cpu'
        )
node = mesh.entity('node')
kwargs = bm.context(node)
idx = bm.arange(10, **kwargs)
idx1 = bm.arange(10, dtype=bm.int32, device='cpu')
print(f"idx: {idx}")
