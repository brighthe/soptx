# 移植自 brighthe/fealpy ``fealpy/mesh/vtk_writter.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""把网格 (选定的实体分区) 写为 VTU 文件.

vtk 只在调用 ``write_mesh_to_vtu`` 时才导入.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib
import json
from typing import Any

import numpy as np

from .storage import MeshBlock
from .view import Mesh

__all__ = [
	"SCHEMA_TO_VTK_CELL_TYPE_NAME",
	"write_mesh_to_vtu",
]


# 映射表按常量名保存, 免得导入本模块时就导入 vtk.
SCHEMA_TO_VTK_CELL_TYPE_NAME: dict[str, str] = {
	"node": "VTK_VERTEX",
	"edge": "VTK_LINE",
	"tri": "VTK_TRIANGLE",
	"quad": "VTK_QUAD",
	"tet": "VTK_TETRA",
	"prism": "VTK_WEDGE",
	"pyramid": "VTK_PYRAMID",
	"hex": "VTK_HEXAHEDRON",
}


def _load_vtk():
	try:
		vtk = importlib.import_module("vtk")
		vnp = importlib.import_module("vtk.util.numpy_support")
	except ImportError as exc:
		raise ImportError(
			"Writing VTU files requires the 'vtk' package. "
			"Please install it first (e.g. `pip install vtk`)."
		) from exc
	return vtk, vnp


def _as_numpy(x: Any) -> np.ndarray:
	return np.asarray(x)


def _to_3d_points(points: np.ndarray) -> np.ndarray:
	if points.ndim != 2:
		raise ValueError(f"Expected points with shape (N, GD), got shape {points.shape}.")

	gd = points.shape[1]
	if gd == 3:
		return points
	if gd == 2:
		out = np.zeros((points.shape[0], 3), dtype=points.dtype)
		out[:, :2] = points
		return out
	if gd == 1:
		out = np.zeros((points.shape[0], 3), dtype=points.dtype)
		out[:, 0] = points[:, 0]
		return out

	raise ValueError(f"Unsupported geometric dimension {gd}; expected 1, 2, or 3.")


def _resolve_vtk_cell_type(schema_name: str, vtk_mod) -> int:
	try:
		vtk_name = SCHEMA_TO_VTK_CELL_TYPE_NAME[schema_name]
	except KeyError as exc:
		raise ValueError(
			f"Unsupported schema '{schema_name}' for VTU export. "
			f"Supported schemas: {sorted(SCHEMA_TO_VTK_CELL_TYPE_NAME)}"
		) from exc
	return int(getattr(vtk_mod, vtk_name))


def _iter_block_cells(schema_name: str, indices: np.ndarray) -> Iterable[np.ndarray]:
	if schema_name == "node":
		if indices.ndim == 1:
			for idx in indices:
				yield np.asarray([idx], dtype=np.int64)
			return
		if indices.ndim == 2 and indices.shape[1] == 1:
			for row in indices:
				yield np.asarray(row, dtype=np.int64)
			return
		raise ValueError(
			"Node entity indices must have shape (N,) or (N, 1), "
			f"got shape {indices.shape}."
		)

	if indices.ndim != 2:
		raise ValueError(
			f"Entity '{schema_name}' indices must have shape (N, NV), got {indices.shape}."
		)

	for row in indices:
		yield np.asarray(row, dtype=np.int64)


def _normalize_attributes_array(value: Any, count: int, field_name: str) -> np.ndarray:
	arr = _as_numpy(value)

	if arr.ndim == 0:
		arr = np.full([count,], arr.item())

	if arr.shape[0] != count:
		raise ValueError(
			f"attributes '{field_name}' has incompatible leading dimension: "
			f"expected {count}, got {arr.shape[0]}."
		)

	# VTK 的向量通常为 (N, 3), 二维向量补零.
	if arr.ndim == 2 and arr.shape[1] == 2:
		out = np.zeros((count, 3), dtype=arr.dtype)
		out[:, :2] = arr
		return out

	return arr


def _create_cell_data_array(
	values: np.ndarray,
	total_count: int,
	start: int,
	end: int,
) -> np.ndarray:
	if values.ndim == 1:
		if np.issubdtype(values.dtype, np.floating):
			result = np.full((total_count,), np.nan, dtype=values.dtype)
		else:
			result = np.zeros((total_count,), dtype=values.dtype)
		result[start:end] = values
		return result

	tail_shape = values.shape[1:]
	if np.issubdtype(values.dtype, np.floating):
		result = np.full((total_count, *tail_shape), np.nan, dtype=values.dtype)
	else:
		result = np.zeros((total_count, *tail_shape), dtype=values.dtype)
	result[start:end] = values
	return result


def write_mesh_to_vtu(
	filename: str,
	mesh: Mesh | MeshBlock,
	*,
	entity_names: Iterable[str] | None = None,
	binary: bool = True,
) -> None:
	"""把网格 (选定的实体分区) 写为 VTU 文件.

	Parameters
	----------
	filename : str
	    输出的 ``.vtu`` 文件路径.
	mesh : Mesh or MeshBlock
	    网格块或其视图.
	entity_names : iterable of str, optional
	    要导出的 schema 名, 如 ``["tri", "edge"]``; 省略时导出 ``mesh.block`` 的
	    全部实体分区.
	binary : bool, optional
	    为 True (默认) 时写二进制 VTU, 否则写 ASCII VTU.

	Raises
	------
	ValueError
	    ``mesh`` 既不是 ``Mesh`` 也不是 ``MeshBlock``.
	"""
	vtk, vnp = _load_vtk()
	if isinstance(mesh, Mesh):
		block = mesh.block
	elif isinstance(mesh, MeshBlock):
		block = mesh
	else:
		raise ValueError("Input mesh must be a Mesh or MeshBlock instance, "
	   		f"got {type(mesh)!r}.")

	points = _to_3d_points(_as_numpy(block.positions))
	vtk_points = vtk.vtkPoints()
	vtk_points.SetData(vnp.numpy_to_vtk(points))

	grid = vtk.vtkUnstructuredGrid()
	grid.SetPoints(vtk_points)

	if entity_names is None:
		selected = list(block.sectors.keys())
	else:
		selected = list(entity_names)

	unknown = [name for name in selected if not block.has_sector(name)]
	if unknown:
		raise ValueError(f"Unknown entity names for export: {unknown}")

	total_cells = 0
	cell_attributes_records: list[tuple[str, str, int, int, Any]] = []
	point_attributes_records: list[tuple[str, Any]] = []

	for schema_name in selected:
		sector = block.get_sector(schema_name)
		indices = _as_numpy(sector.indices)
		cell_schema_name = sector.schema.name
		vtk_cell_type = _resolve_vtk_cell_type(cell_schema_name, vtk)

		start = total_cells
		for node_ids in _iter_block_cells(cell_schema_name, indices):
			grid.InsertNextCell(vtk_cell_type, int(len(node_ids)), node_ids)
			total_cells += 1
		end = total_cells

		for key, value in sector.attributes.items():
			if value is None:
				continue
			if cell_schema_name == "node":
				point_attributes_records.append((key, value))
			else:
				cell_attributes_records.append((cell_schema_name, key, start, end, value))

	metadata = [
		{
			"id": sector.id,
			"schema_id": sector.schema.id,
			"root": sector.id in block.root_cell_sector_ids,
		}
		for sector in (block.get_sector(name) for name in selected)
	]
	metadata_array = vtk.vtkStringArray()
	metadata_array.SetName("FEALPY_MESH_META")
	metadata_array.InsertNextValue(json.dumps(metadata, separators=(",", ":")))
	grid.GetFieldData().AddArray(metadata_array)

	cell_data = grid.GetCellData()
	for schema_name, key, start, end, value in cell_attributes_records:
		field_name = key
		values = _normalize_attributes_array(value, end - start, field_name)
		filled = _create_cell_data_array(values, total_cells, start, end)

		if filled.dtype == np.bool_:
			vtk_arr = vnp.numpy_to_vtk(filled.astype(np.int_))
		else:
			vtk_arr = vnp.numpy_to_vtk(filled)
		vtk_arr.SetName(field_name)
		cell_data.AddArray(vtk_arr)

	point_data = grid.GetPointData()
	point_count = points.shape[0]
	for key, value in point_attributes_records:
		field_name = key
		normalized = _normalize_attributes_array(value, point_count, field_name)
		if normalized.dtype == np.bool_:
			vtk_arr = vnp.numpy_to_vtk(normalized.astype(np.int_))
		else:
			vtk_arr = vnp.numpy_to_vtk(normalized)
		vtk_arr.SetName(field_name)
		point_data.AddArray(vtk_arr)

	writer = vtk.vtkXMLUnstructuredGridWriter()
	writer.SetFileName(filename)
	if binary:
		writer.SetDataModeToBinary()
	else:
		writer.SetDataModeToAscii()
	writer.SetInputData(grid)
	writer.Write()
