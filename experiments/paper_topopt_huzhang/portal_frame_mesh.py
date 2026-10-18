"""门式框架的 Gmsh 半域剖分与全域镜像, 供优化和冻结再分析复用."""

from __future__ import annotations

from typing import Any

import gmsh
import numpy as np

from soptx.backend import backend_manager as bm
from soptx.mesh import TriangleMesh
from soptx.problems import PortalFramePatch2d

def build_mesh(problem: PortalFramePatch2d, h: float, *, algorithm: int = 6) -> Any:
    """生成关于 x = L/2 严格对称的全域非结构三角形网格.

    gmsh 在左半域五边形上剖分, 顶边载荷区与其余段用 transfinite 均匀剖分使载荷区端点落在节点上,
    再镜像到右半; 对称面上的节点被两半共用.

    Parameters
    ----------
    problem : PortalFramePatch2d
        门式框架问题, 提供几何与载荷区位置.
    h : float, optional
        目标单元尺寸, 须为 0.5/k 形式以整除顶边分段, 由运行脚本传入.
    algorithm : int, optional
        Gmsh 网格算法编号, 默认 6 (Frontal-Delaunay).

    Returns
    -------
    mesh : TriangleMesh
        带 ``meshdata`` 元数据的全域网格.

    Raises
    ------
    ValueError
        h 不能整除顶边分段.
    RuntimeError
        gmsh 生成了线性三角形以外的单元.
    """
    x_mid, x_patch = problem.symmetry_axis, problem.traction_patch[0]
    for name, length in (("l/2", x_mid - x_patch), ("L/2 - l/2", x_patch)):
        ratio = length / h
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"h={h} 不能整除顶边分段 {name}={length}; 请取 0.5/k 形式的 h, 如 0.25, 0.125, 0.1.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("portal_half")
        geo = gmsh.model.geo
        points = [geo.addPoint(x, y, 0.0, h) for x, y in
                  ((0.0, 0.0), (problem.b, 0.0), (x_mid, problem.h_c),
                   (x_mid, problem.H), (x_patch, problem.H), (0.0, problem.H))]
        # 边依次为: 支座段, 斜边, 对称面, 顶边载荷区, 顶边其余, 左边
        lines = [geo.addLine(points[i], points[(i + 1) % 6]) for i in range(6)]
        geo.addPlaneSurface([geo.addCurveLoop(lines)])
        geo.synchronize()
        gmsh.model.mesh.setTransfiniteCurve(lines[3], int(round((x_mid - x_patch) / h)) + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[4], int(round(x_patch / h)) + 1)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[:, :2]
        types, _, node_tags = gmsh.model.mesh.getElements(2)
        if list(types) != [2]:
            raise RuntimeError(f"期望只有线性三角形 (gmsh type 2), 实际 {list(types)}")
        tri = np.asarray(node_tags[0], dtype=np.int64).reshape(-1, 3)
    finally:
        gmsh.finalize()

    # gmsh 节点标签不保证连续: 映射到 0 基编号并去掉未被单元引用的节点
    index = {int(tag): i for i, tag in enumerate(tags)}
    cell_half = np.vectorize(index.__getitem__)(tri)
    used = np.unique(cell_half)
    remap = -np.ones(coords.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    node_half, cell_half = coords[used], remap[cell_half]

    # 对称面上的节点横坐标钉到精确值, 镜像合并时按精确相等判定
    node_half[np.abs(node_half[:, 0] - x_mid) < 1e-8 * problem.L, 0] = x_mid

    # 半域单元统一为逆时针
    p0, p1, p2 = node_half[cell_half[:, 0]], node_half[cell_half[:, 1]], node_half[cell_half[:, 2]]
    flip = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]) < 0
    cell_half[flip] = cell_half[flip][:, [0, 2, 1]]

    # 镜像: 对称面上的节点被两半共用, 镜像单元翻转顶点顺序以保持逆时针; 左半编号在前, 右半在后
    off_axis = node_half[:, 0] != x_mid
    new_index = np.arange(node_half.shape[0])
    new_index[off_axis] = node_half.shape[0] + np.arange(int(off_axis.sum()))
    node_right = np.column_stack((2.0 * x_mid - node_half[off_axis, 0], node_half[off_axis, 1]))
    mesh: Any = TriangleMesh(bm.array(np.vstack((node_half, node_right))),
                             bm.array(np.vstack((cell_half, new_index[cell_half][:, [0, 2, 1]]))))
    mesh.meshdata = {"domain": list(problem.domain), "mesh_type": "triangle-gmsh-mirrored", "h": h,
                     "symmetry_axis": x_mid}
    return mesh
