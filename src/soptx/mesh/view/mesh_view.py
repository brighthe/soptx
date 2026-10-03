# 移植自 brighthe/fealpy ``fealpy/mesh/view/mesh_view.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Root-cell-anchored single-block mesh views.

``MeshView`` is the standard single-``MeshBlock`` computational context.  The
classic ``shape_function``/``grad_shape_function`` methods intentionally retain
their historical call convention and delegate to private ``EntityView`` basis
adapters; those methods are part of the classic compatibility contract, not
the general FunctionSpace protocol.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, overload, Self

from ...backend import bm, Tensor
from ..schema import registry as _Reg
from ..storage import MeshBlock
from .entity_view import EntityView


class AdjointRelation(NamedTuple):
    src_indices: Tensor | None
    tgt_indices: Tensor


@dataclass
class MeshView:
    """Provides a root-cell-anchored view of one mesh block.

    A view is anchored to one ``root_cell_sector_id``.  Its topological
    dimension and role resolution come from that anchor rather than from the
    highest dimension found across the whole block.

    Attributes:
        block: The discrete storage block shared by this view.
        cell_sector_id: The root cell sector id used as the anchor.  When
            ``None`` and the block has exactly one root cell sector, that
            sector is inferred; a multi-root block requires an explicit value.
    """
    block: MeshBlock
    cell_sector_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the supplied or inferred anchor sector id."""
        if (
            self.cell_sector_id is not None
            and self.cell_sector_id not in self.block.root_cell_sector_ids
        ):
            raise ValueError(
                f"cell_sector_id {self.cell_sector_id!r} is not a root "
                "cell sector"
            )
        if (
            self.cell_sector_id is None
            and len(self.block.root_cell_sector_ids) > 1
        ):
            raise ValueError(
                "MeshView requires an explicit cell_sector_id for a "
                "multi-root MeshBlock"
            )

    def _anchor_sector_id(self) -> str | None:
        """Return the single root cell sector id for this view."""
        if self.cell_sector_id is not None:
            if self.cell_sector_id not in self.block.root_cell_sector_ids:
                raise ValueError(
                    f"cell_sector_id {self.cell_sector_id!r} is not a root "
                    "cell sector"
                )
            return self.cell_sector_id

        roots = self.block.root_cell_sector_ids
        if len(roots) == 1:
            return roots[0]
        if len(roots) > 1:
            raise ValueError(
                "MeshView requires an explicit cell_sector_id for a "
                "multi-root MeshBlock"
            )
        return None

    def _visible_sector_ids(self) -> list[str]:
        """Return the logical closure of sector ids visible from this view."""
        ids: list[str] = []
        if "node" in self.block.sectors:
            ids.append("node")

        anchor = self._anchor_sector_id()
        if anchor is not None:
            ids.append(anchor)
            ids.extend(
                sector.id
                for sector in self.block.sectors.values()
                if sector.source_cell_sector_id == anchor
                and sector.id != anchor
            )
        else:
            ids.extend(
                sector.id
                for sector in self.block.sectors.values()
                if sector.id != "node"
            )
        return list(dict.fromkeys(ids))

    def _entity_name(self, name_or_topdim: str | int, idx: int) -> str:
        """Resolve a legacy entity selector within the view's closure."""
        visible = self._visible_sector_ids()
        if isinstance(name_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(
                name_or_topdim,
                self.top_dimension(),
            )
        elif isinstance(name_or_topdim, str):
            if ":" in name_or_topdim:
                topdim, idx = _Reg.string_to_etype_and_idx(
                    name_or_topdim,
                    self.top_dimension(),
                )
            elif name_or_topdim in visible:
                return name_or_topdim
            else:
                topdim = _Reg.etype_to_topdim(
                    name_or_topdim,
                    self.top_dimension(),
                )
        else:
            raise TypeError(
                f"Expected str or int for name_or_topdim, "
                f"got {type(name_or_topdim)}"
            )

        names = [
            name
            for name in visible
            if self.block.get_sector(name).schema.top_dim == topdim
        ]
        return names[idx]

    ## Entity getters

    def entity_views(self, etype_or_topdim: str | int, /) -> list[EntityView]:
        """Return all entity views for a role or topological dimension.

        Unlike :meth:`entity_view`, this method never raises on ambiguity and
        returns every sector matching the selector.
        """
        visible = self._visible_sector_ids()
        if isinstance(etype_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(
                etype_or_topdim,
                self.top_dimension(),
            )
        else:
            if not isinstance(etype_or_topdim, str):
                raise TypeError(
                    "Expected str or int for etype_or_topdim, "
                    f"got {type(etype_or_topdim).__name__}"
                )
            if etype_or_topdim in visible:
                return [EntityView(self.block, etype_or_topdim)]
            topdim = _Reg.etype_to_topdim(
                etype_or_topdim,
                self.top_dimension(),
            )

        names = [
            name
            for name in visible
            if self.block.get_sector(name).schema.top_dim == topdim
        ]
        return [EntityView(self.block, name) for name in names]

    def entity_view(self, name_or_topdim: str | int, /) -> EntityView:
        """Return one :class:`EntityView` for an unambiguous entity selector.

        A concrete sector id is returned directly.  A role such as ``"cell"``,
        ``"face"``, ``"edge"`` or ``"node"`` is resolved against the current
        anchor.  When one role maps to multiple sectors, the method raises
        rather than silently selecting one of them.

        Parameters:
            name_or_topdim: A concrete sector id, a role name, or an integer
                topological dimension relative to this anchor.

        Returns:
            The selected :class:`EntityView`.

        Raises:
            TypeError: If the selector has an unsupported type.
            ValueError: If the role is unknown or maps to zero or multiple
                sectors in this view.
        """
        visible = self._visible_sector_ids()
        if isinstance(name_or_topdim, str) and name_or_topdim in visible:
            return EntityView(self.block, name_or_topdim)

        if isinstance(name_or_topdim, int):
            topdim = _Reg.ensure_positive_topdim(
                name_or_topdim,
                self.top_dimension(),
            )
        elif isinstance(name_or_topdim, str):
            topdim = _Reg.etype_to_topdim(
                name_or_topdim,
                self.top_dimension(),
            )
        else:
            raise TypeError(
                f"Expected str or int for name_or_topdim, "
                f"got {type(name_or_topdim)}"
            )

        names = [
            name
            for name in visible
            if self.block.get_sector(name).schema.top_dim == topdim
        ]
        if not names:
            raise ValueError(
                f"no entity sector resolves role {name_or_topdim!r} "
                "in this view"
            )
        if len(names) > 1:
            raise ValueError(
                f"role {name_or_topdim!r} is ambiguous in this view; "
                f"available sectors: {names}"
            )
        return EntityView(self.block, names[0])

    def entity(self, name_or_topdim: str | int, /) -> Tensor:
        """Return entity data in the classical FEALPy layout.

        The node role returns the coordinate tensor; every other entity role
        returns its integer connectivity array.

        Parameters:
            name_or_topdim: A concrete sector id, a role name, or an integer
                topological dimension relative to this anchor.

        Returns:
            Node coordinates with shape ``(NN, GD)`` for nodes, or integer
            connectivity with shape ``(NE, NVE)`` for other entities.
        """
        if name_or_topdim in {"node", "Node", "NODE", 0}:
            return self.block.positions
        return self.entity_view(name_or_topdim).indices

    @property
    def cell_view(self) -> EntityView:
        """Return the anchored cell :class:`EntityView`."""
        return self.entity_view("cell")

    @property
    def face_view(self) -> EntityView:
        """Return the codimension-one face :class:`EntityView`.

        Raises:
            ValueError: If the face role is ambiguous in this view.
        """
        return self.entity_view("face")

    @property
    def edge_view(self) -> EntityView:
        """Return the edge :class:`EntityView`.

        Raises:
            ValueError: If the edge role is ambiguous in this view.
        """
        return self.entity_view("edge")

    @property
    def node_view(self) -> EntityView:
        """Return the block-global node :class:`EntityView`."""
        return self.entity_view("node")

    @property
    def cell(self) -> Tensor:
        """Return cell connectivity data."""
        return self.entity("cell")

    @property
    def face(self) -> Tensor:
        """Return face connectivity data."""
        return self.entity("face")

    @property
    def edge(self) -> Tensor:
        """Return edge connectivity data."""
        return self.entity("edge")

    @property
    def node(self) -> Tensor:
        """Return the node coordinate data."""
        return self.entity("node")

    ## Checkers

    def is_simplex_mesh(self) -> bool:
        """Check if the mesh is a simplex mesh."""
        for name in self._visible_sector_ids():
            schema_name = self.block.get_sector(name).schema.name
            if schema_name not in {"node", "edge", "tri", "tet"}:
                return False
        return True

    def is_tensor_mesh(self) -> bool:
        """Check if the mesh is a tensor mesh."""
        for name in self._visible_sector_ids():
            schema_name = self.block.get_sector(name).schema.name
            if schema_name not in {"node", "edge", "quad", "hex"}:
                return False
        return True

    def is_elemental(self, entity_name: str | None = None, /) -> bool:
        """Check if the mesh has only one root entity type."""
        anchor = self._anchor_sector_id()
        is_one_root = anchor is not None
        if entity_name is None:
            return is_one_root
        return is_one_root and anchor == entity_name

    ## Other getters

    def geo_dimension(self) -> int:
        """Get the geometric dimension of the mesh."""
        return int(self.block.positions.shape[1])

    def top_dimension(self) -> int:
        """Return the topological dimension from the anchor cell Schema.

        Returns:
            The anchor cell's topological dimension, or ``-1`` when this view
            has no root cell anchor (for example a node-only legacy block).
        """
        anchor = self._anchor_sector_id()
        if anchor is None:
            return -1
        return self.block.get_sector(anchor).schema.top_dim

    @property
    def ftype(self):
        """Return the floating dtype of node coordinates."""
        return self.block.positions.dtype

    @property
    def itype(self):
        """Return the integer dtype used by the canonical node sector."""
        node = self.block.sectors.get("node")
        if node is not None:
            return node.indices.dtype
        return bm.int32

    @property
    def device(self):
        """Return the backend device of node coordinates, if available."""
        return getattr(self.block.positions, "device", None)

    # [Shape functions]

    def shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """Evaluate cell shape functions through the transitional basis API."""
        return self.entity_views(-1)[0]._legacy_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    cell_shape_function = shape_function

    def face_shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """Evaluate face shape functions through the transitional basis API."""
        return self.entity_views(-2)[0]._legacy_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def edge_shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """Evaluate edge shape functions through the transitional basis API."""
        return self.entity_views(1)[0]._legacy_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def grad_shape_function(
        self,
        bcs,
        p=1,
        *,
        index=None,
        variables="u",
        mi=None,
    ):
        """Evaluate cell shape-function gradients through the transitional basis API."""
        return self.entity_views(-1)[0]._legacy_grad_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    # [Counts and interpolation]

    def number_of_cells(self) -> int:
        """Return the total number of top-dimensional entities."""
        return sum(sec.size() for sec in self.entity_views(-1))

    def number_of_faces(self) -> int:
        """Return the total number of codimension-one entities."""
        return sum(sec.size() for sec in self.entity_views(-2))

    def number_of_edges(self) -> int:
        """Return the total number of edge entities."""
        return sum(sec.size() for sec in self.entity_views(1))

    def number_of_nodes(self) -> int:
        """Return the total number of node entities."""
        return sum(sec.size() for sec in self.entity_views(0))

    def number_of_global_ipoints(self, p) -> int:
        """Return the number of global interpolation points of order ``p``."""
        total = 0
        for name in self._visible_sector_ids():
            sector_view = self.entity_view(name)
            if name == "node":
                cell_view = self.cell_view
                local_vertices = cell_view.schema.local_vertices()
                vertex_ids = bm.reshape(
                    cell_view.indices[:, local_vertices],
                    (-1,),
                )
                vertex_count = int(bm.unique(vertex_ids).shape[0])
                total += sector_view.num_multi_index(
                    p,
                    internal=True,
                ) * vertex_count
            else:
                total += sector_view.num_multi_index(
                    p,
                    internal=True,
                ) * sector_view.size()
        return total

    def number_of_local_ipoints(self, p, iptype="cell") -> int:
        """Return the number of local interpolation points on one entity."""
        return self.entity_view(iptype).num_multi_index(p)

    def multi_index_matrix(self, p, etype="cell"):
        """Return the local multi-index matrix for an entity type."""
        return self.entity_view(etype).multi_index_matrix(p)

    def quadrature_formula(self, q, name_or_topdim="cell", qtype="legendre"):
        """Return a quadrature formula for the selected entity type."""
        return self.entity_view(name_or_topdim).quadrature_formula(q, qtype)

    @property
    def localEdge(self):
        """Return the local edge-to-vertex table of the cell reference entity."""
        cell_sec = self.entity_views(-1)[0]
        edge_sec = self.entity_views(1)[0]
        data = cell_sec.schema.local_entity(edge_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    @property
    def localFace(self):
        """Return the local face-to-vertex table of the cell reference entity."""
        cell_sec = self.entity_views(-1)[0]
        face_sec = self.entity_views(-2)[0]
        data = cell_sec.schema.local_entity(face_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    def cell_to_ipoint(self, p, *, index=None):
        """Map cells to global interpolation-point indices."""
        from ..ipoints import to_ipoint
        view = self.entity_views(-1)[0]
        result = to_ipoint(self, view, p)
        return result if index is None else result[index]

    def face_to_ipoint(self, p, *, index=None):
        """Map faces to global interpolation-point indices."""
        from ..ipoints import to_ipoint
        view = self.entity_views(-2)[0]
        result = to_ipoint(self, view, p)
        return result if index is None else result[index]

    def edge_to_ipoint(self, p, *, index=None):
        """Map edges to global interpolation-point indices."""
        from ..ipoints import to_ipoint
        view = self.edge_view
        result = to_ipoint(self, view, p)
        return result if index is None else result[index]

    def interpolation_points(self, p, entity=None, index=None):
        """Return coordinates of global interpolation points."""
        from ..ipoints import ipoints

        visible = self._visible_sector_ids()
        names: list[str] = []

        def _sector_ids_by_topdim(top_dim: int) -> list[str]:
            return [
                name
                for name in visible
                if self.block.get_sector(name).schema.top_dim == top_dim
            ]

        if entity is None:
            for top_dim in range(self.top_dimension() + 1):
                names.extend(_sector_ids_by_topdim(top_dim))
        else:
            selectors = (
                list(entity)
                if isinstance(entity, Iterable) and not isinstance(entity, str)
                else [entity]
            )
            for selector in selectors:
                if isinstance(selector, int):
                    top_dim = _Reg.ensure_positive_topdim(
                        selector,
                        self.top_dimension(),
                    )
                    names.extend(_sector_ids_by_topdim(top_dim))
                else:
                    names.append(self.entity_view(selector).sector_id)

        ips = ipoints(self, p, names)
        return ips if index is None else ips[index, :]

    # [Topology]

    def cell_to_face(self, src_id=0, dst_id=0):
        """Return the cell-to-face connectivity map."""
        cell = self.entity_views("cell")[src_id]
        face = self.entity_views("face")[dst_id]
        return cell.to(face).as_array()

    def cell_to_edge(self, src_id=0, dst_id=0):
        """Return the cell-to-edge connectivity map."""
        cell = self.entity_views("cell")[src_id]
        edge = self.entity_views("edge")[dst_id]
        return cell.to(edge).as_array()

    def face_to_edge(self, src_id=0, dst_id=0):
        """Return the face-to-edge connectivity map."""
        face = self.entity_views("face")[src_id]
        edge = self.entity_views("edge")[dst_id]
        return face.to(edge).as_array()

    def face_to_cell(self, src_id=0, dst_id=0):
        """Return sparse face-to-cell adjacency."""
        face = self.entity_views("face")[src_id]
        cell = self.entity_views("cell")[dst_id]
        rel = face.to(cell)
        data = rel.unique
        lidx = rel.local_index
        return bm.stack([data.first, data.last, lidx.floc, lidx.lloc], axis=1)

    def cell_to_edge_sign(self):
        """Return orientation signs of local cell edges."""
        cell_sec = self.entity_views(-1)[0]
        edge_sec = self.entity_views(1)[0]
        c2e = self.cell_to_edge()
        local_pair = cell_sec.indices[:, self.localEdge]
        global_pair = edge_sec.indices[c2e]
        return bm.all(local_pair == global_pair, axis=-1)

    def face_to_edge_sign(self):
        """Return orientation signs of local face edges."""
        face_sec = self.entity_views(-2)[0]
        edge_sec = self.entity_views(1)[0]
        f2e = self.face_to_edge()
        sign = bm.zeros((face_sec.indices.shape[0], 3), dtype=bm.bool)
        local_f2e = face_sec.schema.local_entity("edge")
        n = [item[0] for item in local_f2e]
        for i in range(len(n)):
            sign[:, i] = face_sec.indices[:, n[i]] == edge_sec.indices[f2e[:, i], 0]
        return sign

    def boundary_cell_flag(self):
        """Return a boolean mask marking boundary cells."""
        return self.entity_view(-1).boundary().mask

    def boundary_face_flag(self):
        """Return a boolean mask marking boundary faces."""
        return self.entity_view(-2).boundary().mask

    def boundary_edge_flag(self):
        """Return a boolean mask marking boundary edges."""
        return self.entity_view(1).boundary().mask

    def boundary_node_flag(self):
        """Return a boolean mask marking boundary nodes."""
        return self.entity_view(0).boundary().mask

    def boundary_cell_index(self):
        """Return global indices of boundary cells."""
        return self.entity_view(-1).boundary().index

    def boundary_face_index(self):
        """Return global indices of boundary faces."""
        return self.entity_view(-2).boundary().index

    def boundary_edge_index(self):
        """Return global indices of boundary edges."""
        return self.entity_view(1).boundary().index

    def boundary_node_index(self):
        """Return global indices of boundary nodes."""
        return self.entity_view(0).boundary().index

    # [Geometry]

    def bc_to_point(self, bc, *, index=None):
        """Convert barycentric coordinates to physical points."""
        if not isinstance(bc, tuple):
            bc = (bc,)
        top = sum(b.shape[1] - 1 for b in bc)
        return self.entity_views(top)[0].bc_to_point(bc, index=index)

    def entity_barycenter(self, name_or_topdim, /, *, index=None):
        """Return barycenters of selected entities."""
        return self.entity_view(name_or_topdim).barycenter(index=index)

    def entity_measure(self, name_or_topdim, /, *, index=None):
        """Return measures of selected entities."""
        return self.entity_view(name_or_topdim).measure(index=index)

    def edge_tangent(self, *, index=None):
        """Return edge tangent vectors."""
        return self.entity_views(1)[0].tangent(index=index)[:, 0, :]

    def edge_unit_tangent(self, *, index=None):
        """Return unit tangent vectors of edges."""
        tangent = self.entity_views(1)[0].tangent(index=index)[:, 0, :]
        norm = bm.linalg.vector_norm(tangent, axis=1, keepdims=True)
        return tangent / norm

    def face_normal(self, *, index=None):
        """Return face normal vectors."""
        return self.entity_views(self.top_dimension() - 1)[0].normal(index=index)[:, 0, :]

    def face_unit_normal(self, *, index=None):
        """Return unit normal vectors of faces."""
        normal = self.entity_views(self.top_dimension() - 1)[0].normal(index=index)[:, 0, :]
        norm = bm.linalg.vector_norm(normal, axis=1, keepdims=True)
        return normal / norm

    def grad_lambda(self, index=None, TD=None):
        """Return gradients of barycentric coordinate functions."""
        if TD is None:
            TD = self.top_dimension()
        if TD == self.top_dimension():
            block = self.entity_views(self.top_dimension())[0]
        elif TD == self.top_dimension() - 1:
            block = self.entity_views(self.top_dimension() - 1)[0]
        elif TD == 1:
            block = self.entity_views(1)[0]
        else:
            raise ValueError(f"Unsupported top dimension: {TD}")
        return block.grad_lambda(index=index)

    def grad_face_lambda(self, index=None):
        """Return gradients of face barycentric coordinate functions."""
        return self.grad_lambda(index=index, TD=self.top_dimension() - 1)

    def error(self, f1, f2, /, power=2.0, q=3, *, cell_axis=False, index=None):
        """Compute the mesh norm of the difference between two functions."""
        return self.entity_views(-1)[0].error(
            f1, f2, power=power, q=q, cell_axis=cell_axis, index=index
        )

    # Setters (in-place modification)

    def construct(self, exclude: list[str] | None = None) -> Self:
        """Construct the mesh topology, optionally excluding certain entity types."""
        from ..topology.builder import TopologyBuilder
        TopologyBuilder.construct(self.block, exclude=exclude)
        return self

    def uniform_refine(self, n: int = 1, **kwargs):
        """Uniformly refine the mesh a given number of times.

        Parameters:
            n: Number of times to refine the mesh. Default is 1.
            **kwargs: Additional arguments for the refinement process.
        """
        if len(self.block.root_cell_sector_ids) > 1:
            raise NotImplementedError(
                "MeshView.uniform_refine() does not support a multi-root "
                "MeshBlock"
            )

        from ..transform import uniform_refine
        return uniform_refine(
            self.block,
            n=n,
            root_id=self._anchor_sector_id(),
            **kwargs,
        )
