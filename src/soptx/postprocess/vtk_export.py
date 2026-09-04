"""将有限元场导出为 ParaView 可读的 VTU 非结构化网格文件.

examples 里 lagrange/huzhang 的 ``export_vtu`` 与 pinn 的内联 VTU 导出逻辑
原为三处重复, 这里集中为统一的底层 ``write_vtu`` 与便捷封装 ``export_vtu``:

* ``write_vtu``: 按网格自动识别单元类型 (2D 三角形/四边形, 3D 四面体),
  调用方提供任意节点场 dict —— 供 pinn 这类导出多个场的场景使用;
* ``export_vtu``: 单个位移场的便捷封装, 输出标准字段 ``u_x/u_y/.../u_mag``
  —— 供 Lagrange 与 Hu--Zhang 问题导出位移场使用.

依赖 ``pyevtk`` (随 soptx 工作树安装). 本模块只在被显式导入时加载, 不会
拖慢 ``import soptx``.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from pyevtk.hl import unstructuredGridToVTK

# VTK 单元类型常量
_VTK_TRIANGLE = 5
_VTK_QUAD = 9
_VTK_TETRA = 10
_VTK_HEXAHEDRON = 12


def _resolve_cell_type(nodes: np.ndarray, cells: np.ndarray) -> int:
    """按单元节点数 (必要时辅以 z 坐标) 推断 VTK 单元类型.

    3 节点是三角形; 4 节点可能是 2D 四边形或 3D 四面体; 8 节点是 3D 六面体.
    """

    nodes_per_cell = cells.shape[1]
    if nodes_per_cell == 3:
        return _VTK_TRIANGLE
    if nodes_per_cell == 4:
        is_3d = nodes.shape[1] >= 3 and np.abs(nodes[:, 2]).max() > 0.0
        return _VTK_TETRA if is_3d else _VTK_QUAD
    if nodes_per_cell == 8:
        return _VTK_HEXAHEDRON
    raise ValueError(f"不支持的单元类型: {nodes_per_cell} 节点")


def write_vtu(
    mesh,
    point_data: dict[str, np.ndarray] | None = None,
    filepath: str = "",
    cell_data: dict[str, np.ndarray] | None = None,
) -> None:
    """把节点场和/或单元场数据写成 VTU 非结构化网格文件.

    参数:
        mesh: FEALPy 网格, 需提供 ``entity('node')`` / ``entity('cell')``.
        point_data: 节点场字典, 值为 ``(n_nodes,)`` 或 ``(n_nodes, gd)`` 数组.
        filepath: 输出路径 (不含扩展名时 pyevtk 自动补 ``.vtu``).
        cell_data: 单元场字典, 值为 ``(n_cells,)`` 或 ``(n_cells, ...)`` 数组.
    """

    nodes = np.asarray(mesh.entity("node"), dtype=np.float64)
    cells = np.asarray(mesh.entity("cell"), dtype=np.int32)

    n_nodes = nodes.shape[0]
    n_cells = cells.shape[0]
    cell_type = _resolve_cell_type(nodes, cells)

    # VTU: 节点坐标分量 (2D 网格补零 z)
    x = np.ascontiguousarray(nodes[:, 0])
    y = np.ascontiguousarray(nodes[:, 1])
    if nodes.shape[1] >= 3:
        z = np.ascontiguousarray(nodes[:, 2])
    else:
        z = np.zeros(n_nodes, dtype=np.float64)

    # 连通性与偏移
    connectivity = np.ascontiguousarray(cells.flatten())
    offsets = np.arange(
        cells.shape[1], n_cells * cells.shape[1] + 1, cells.shape[1], dtype=np.int32
    )
    cell_types = np.full(n_cells, cell_type, dtype=np.int32)

    # 点数据一律转成连续的一维列
    point_data_vtu = None
    if point_data:
        point_data_vtu = {}
        for name, value in point_data.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            point_data_vtu[name] = np.ascontiguousarray(arr)

    # 单元数据
    cell_data_vtu = None
    if cell_data:
        cell_data_vtu = {}
        for name, value in cell_data.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            cell_data_vtu[name] = np.ascontiguousarray(arr)

    unstructuredGridToVTK(
        filepath, x, y, z, connectivity, offsets, cell_types,
        pointData=point_data_vtu,
        cellData=cell_data_vtu,
    )


def read_vtu_cell_data(filepath: Path | str, name: str = "density") -> np.ndarray:
    """从 ``write_vtu`` 写出的 VTU 里读回一个单元场.

    ``pyevtk`` 落盘的是 appended raw 段, float64 原样写入不做文本化, 因此
    读回的值与写入值逐位相同 —— 需要精确复算的对照实验可以直接把 VTU 当
    数据源, 不必再另存一份 ``.npy``.

    参数:
        filepath: VTU 文件路径.
        name: CellData 数组名, 缺省取拓扑优化的 ``density``.

    返回形状 ``(n_cells,)`` 的 float64 数组 (对原始字节做拷贝, 不持有文件缓冲).
    """

    raw = Path(filepath).read_bytes()
    try:
        header = raw[: raw.index(b"<AppendedData")].decode("utf-8", "ignore")
    except ValueError as error:
        raise ValueError(f"{filepath} 不是 appended-raw 格式的 VTU.") from error
    match = re.search(rf'Name="{re.escape(name)}"[^>]*offset="(\d+)"', header)
    if match is None:
        raise ValueError(f"{filepath} 里没有名为 {name!r} 的数据数组.")
    n_cells = int(re.search(r'NumberOfCells="(\d+)"', header).group(1))
    base = raw.index(b"_", raw.index(b'encoding="raw">')) + 1
    offset = base + int(match.group(1))
    # appended 段每个数组前有 8 字节长度头; 对不上说明取到的不是单元场。
    n_bytes = int(np.frombuffer(raw, dtype=np.uint64, count=1, offset=offset)[0])
    if n_bytes != n_cells * 8:
        raise ValueError(
            f"{filepath} 的 {name!r} 长度 {n_bytes} 字节与 {n_cells} 个单元不符."
        )
    return np.frombuffer(raw, dtype=np.float64, count=n_cells, offset=offset + 8).copy()


def export_vtu(mesh, displacement: np.ndarray, filepath: str) -> None:
    """便捷封装: 单个位移场导出为 VTU, 字段为 ``u_x``/``u_y``/.../``u_mag``.

    ``displacement`` 形状 ``(n_nodes,)`` 或 ``(n_nodes, gd)``, 必须是已经过
    节点插值的值 —— 对不连续空间 (如胡张元的位移空间) 调用方需先做跨单元平均.
    """

    disp = np.asarray(displacement, dtype=np.float64)
    if disp.ndim == 1:
        disp = disp.reshape(-1, 1)

    point_data = {}
    for d in range(disp.shape[1]):
        point_data[f"u_{chr(120 + d)}"] = disp[:, d]
    point_data["u_mag"] = np.linalg.norm(disp, axis=1)

    write_vtu(mesh, point_data, filepath)
