"""网格拓扑协调性校验 ``_validate_conforming_occurrences`` 的向量化实现回归测试."""

from __future__ import annotations

import numpy as np
import pytest

from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh, QuadrangleMesh, TetrahedronMesh, TriangleMesh
from soptx.mesh.topology.builder import _validate_conforming_occurrences
from soptx.mesh.topology.local_entity import CanonicalLocalEntityOccurrence


class _SchemaA:
    """只提供校验所需的 ``id``; 校验以 Python 类型区分拓扑键."""

    def __init__(self, schema_id):
        self.id = schema_id


class _SchemaB(_SchemaA):
    pass


def _occurrence(schema, vertices, full_nodes):
    vertices, full_nodes = np.asarray(vertices), np.asarray(full_nodes)
    zeros = np.zeros_like(vertices)
    return CanonicalLocalEntityOccurrence(schema=schema, indices=full_nodes, canonical_vertices=vertices,
                                          vertex_permutation=zeros, node_permutation=zeros)


def _reference(canonical_by_schema):
    """逐个出现循环的参考实现, 与向量化之前的写法相同; 返回 None 或异常类型."""
    seen = {}
    for schema, canonical in canonical_by_schema.items():
        for vertex_row, full_row in zip(np.asarray(canonical.canonical_vertices).tolist(),
                                        np.asarray(canonical.indices).tolist()):
            key = (type(schema), tuple(vertex_row))
            if key not in seen:
                seen[key] = (schema.id, tuple(full_row))
                continue
            if schema.id != seen[key][0]:
                return NotImplementedError
            if tuple(full_row) != seen[key][1]:
                return ValueError
    return None


def _outcome(canonical_by_schema):
    try:
        _validate_conforming_occurrences(canonical_by_schema)
    except (NotImplementedError, ValueError) as error:
        return type(error)
    return None


def _random_conforming(rng, n_entities, n_occurrences, n_vertices, n_full):
    """随机生成协调的出现: 每个实体一组顶点行与完整节点行, 出现为实体的随机重复."""
    vertices = rng.choice(10 * n_entities, size=(n_entities, n_vertices), replace=True)
    vertices = np.unique(vertices, axis=0)
    full = np.concatenate([vertices, rng.integers(0, 10 ** 6, size=(len(vertices), n_full - n_vertices))],
                          axis=1)
    picks = rng.integers(0, len(vertices), size=n_occurrences)
    return vertices[picks], full[picks]


def test_conforming_random_occurrences_pass() -> None:
    bm.set_backend('numpy')
    rng = np.random.default_rng(0)
    vertices, full = _random_conforming(rng, 200, 2000, 2, 3)
    schema = _SchemaA('edge-p2')
    canonical = {schema: _occurrence(schema, vertices, full)}

    assert _reference(canonical) is None
    assert _outcome(canonical) is None


def test_different_full_nodes_raise_value_error() -> None:
    bm.set_backend('numpy')
    rng = np.random.default_rng(1)
    vertices, full = _random_conforming(rng, 200, 2000, 2, 3)
    full = full.copy()
    full[-1, 2] += 1                     # 与同顶点的其它出现中间节点不同
    schema = _SchemaA('edge-p2')
    canonical = {schema: _occurrence(schema, vertices, full)}

    assert _reference(canonical) is ValueError
    assert _outcome(canonical) is ValueError


def test_schema_id_mismatch_raises_not_implemented() -> None:
    bm.set_backend('numpy')
    p1, p2 = _SchemaA('edge-p1'), _SchemaA('edge-p2')
    canonical = {p1: _occurrence(p1, [[0, 1], [1, 2]], [[0, 1], [1, 2]]),
                 p2: _occurrence(p2, [[1, 2], [5, 6]], [[1, 2], [5, 6]])}   # 顶点行 (1, 2) 共享

    assert _reference(canonical) is NotImplementedError
    assert _outcome(canonical) is NotImplementedError


def test_distinct_schema_types_and_vertex_counts_do_not_collide() -> None:
    bm.set_backend('numpy')
    a, b, c = _SchemaA('a'), _SchemaB('b'), _SchemaA('a3')
    canonical = {a: _occurrence(a, [[0, 1], [0, 1]], [[0, 1], [0, 1]]),
                 b: _occurrence(b, [[0, 1]], [[0, 9]]),                    # 类型不同, 完整节点可不同
                 c: _occurrence(c, [[0, 1, 2]], [[0, 1, 2]])}              # 同类型但顶点数不同

    assert _reference(canonical) is None
    assert _outcome(canonical) is None


@pytest.mark.parametrize('seed', range(20))
def test_vectorized_matches_reference_on_perturbed_inputs(seed) -> None:
    bm.set_backend('numpy')
    rng = np.random.default_rng(seed)
    vertices, full = _random_conforming(rng, 50, 300, 3, 4)
    full = full.copy()
    # 每个种子只制造一种冲突: 两种同时出现时参考实现按遍历顺序报错, 与向量化的固定顺序不可比
    id_conflict = seed % 3 == 0
    if not id_conflict and seed % 2:
        row = rng.integers(0, len(full))
        full[row, 3] += 1
    schemas = [_SchemaA('tri-p2'), _SchemaA('tri-p2b' if id_conflict else 'tri-p2')]
    split = len(vertices) // 2
    canonical = {schemas[0]: _occurrence(schemas[0], vertices[:split], full[:split]),
                 schemas[1]: _occurrence(schemas[1], vertices[split:], full[split:])}

    assert _outcome(canonical) is _reference(canonical)


@pytest.mark.parametrize('mesh_class, shape, expected', [
    (TriangleMesh, (4, 3), {'cell': 24, 'edge': 4 * 4 + 5 * 3 + 4 * 3 * 1}),
    (QuadrangleMesh, (4, 3), {'cell': 12, 'edge': 4 * 4 + 5 * 3}),
    (HexahedronMesh, (4, 3, 2), {'cell': 24, 'face': 3 * 24 + 4 * 3 + 3 * 2 + 4 * 2,
                                 'edge': 4 * 4 * 3 + 5 * 3 * 3 + 5 * 4 * 2}),
    (TetrahedronMesh, (2, 2, 2), {'cell': 48}),
])
def test_box_meshes_construct_with_expected_entity_counts(mesh_class, shape, expected) -> None:
    bm.set_backend('numpy')
    box = [0.0, 1.0] * len(shape)
    mesh = mesh_class.from_box(box, *shape)

    for entity, count in expected.items():
        assert getattr(mesh, f'number_of_{entity}s')() == count
