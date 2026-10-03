# 移植自 brighthe/fealpy ``fealpy/mesh/schema/entity_schema.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Contracts for immutable mesh-entity Schema values.

An :class:`EntitySchema` value describes one concrete reference entity: its
stable identity, complete local-node layout, admissible orientations, geometry
interpolation, and optional reference-basis capabilities.  Schema values do
not own mesh connectivity, physical coordinates, or finite-element DoFs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import ClassVar, Concatenate, Literal, ParamSpec, TYPE_CHECKING, cast

from ...backend import Tensor, Index
from ..storage import EntityContext
from .descriptor import CanonicalValue, SchemaDescriptor

if TYPE_CHECKING:
    from ...quadrature import Quadrature
    from .local_entity import LocalEntityGroup

__all__ = ["EntitySchema"]

P = ParamSpec("P")


class EntitySchema:
    """Define one immutable, parameterized reference-entity contract.

    A Schema Python class identifies a structural family, while each instance
    identifies one normalized member of that family.  For example,
    ``LagrangeTriangleSchema(p=1)`` and ``LagrangeTriangleSchema(p=2)`` have
    the same Python type but different descriptors, IDs, local-node layouts,
    and geometry bases.

    Concrete Schema values are immutable and compare by value.  Callers must
    use equality, :attr:`descriptor`, or :attr:`id` for identity and must not
    rely on Python object identity.  A Schema contains no ``MeshBlock`` state
    and no backend, device, or dtype-specific tensor cache.
    """

    __slots__ = ()

    type_id: ClassVar[str]
    schema_version: ClassVar[int]
    descriptor_parameter_names: ClassVar[tuple[str, ...]]
    name: ClassVar[str]
    top_dim: ClassVar[int]
    OFace: ClassVar[Mapping[str, tuple[tuple[int, ...], ...]]] = MappingProxyType({})
    SFace: ClassVar[Mapping[str, tuple[tuple[int, ...], ...]]] = MappingProxyType({})
    orientation: ClassVar[tuple[tuple[int, ...], ...]] = ()
    ccw: ClassVar[tuple[int, ...] | None] = None
    ref_measure: ClassVar[float | None] = None

    @property
    def descriptor(self) -> SchemaDescriptor:
        """Return the canonical descriptor derived from normalized fields.

        Parameter names and order follow ``descriptor_parameter_names`` on
        the registered Schema type.  The returned value is immutable and is
        sufficient to reconstruct an equal Schema through ``SCHEMA_RESOLVER``.
        """
        parameters = tuple(
            (name, cast(CanonicalValue, getattr(self, name)))
            for name in self.descriptor_parameter_names
        )
        return SchemaDescriptor(
            type_id=self.type_id,
            schema_version=self.schema_version,
            parameters=parameters,
        )

    @property
    def id(self) -> str:
        """Return the deterministic, reversible ID of this concrete Schema.

        The ID encodes :attr:`descriptor`; it is independent of module paths,
        process-local registry insertion order, and Python object identity.
        """
        return self.descriptor.to_id()

    def number_of_vertices(self) -> int:
        """Return the size of the topological vertex skeleton.

        This count is independent of interpolation order and can be smaller
        than :meth:`number_of_nodes` for a high-order geometry Schema.
        """
        raise NotImplementedError()

    def number_of_nodes(self) -> int:
        """Return the width of this Schema's complete local-node layout.

        The layout contains vertices and all higher-order interpolation nodes
        used by the geometry basis.  It is the required connectivity width of
        an entity sector bound to this concrete Schema.
        """
        raise NotImplementedError()

    def validate_connectivity_counts(self, counts: Tensor) -> None:
        """Validate entity-wise connectivity counts for a ragged sector.

        Fixed-cardinality Schemas keep the transitional permissive behavior
        for existing variable-sector callers. Variable-cardinality Schemas
        override this hook with their structural cardinality contract.
        """

    def local_vertices(self) -> tuple[int, ...]:
        """Return vertex-column positions in the complete local-node layout.

        The result indexes local connectivity columns; it does not contain
        block-global node IDs.
        """
        raise NotImplementedError()

    def local_entity_groups(self, top_dim: int) -> tuple[LocalEntityGroup, ...]:
        """Describe local subentities of one topological dimension.

        Each returned group binds an immutable child Schema to rows of parent
        local-node column indices.  Every row follows the child's complete
        canonical node order.  Multiple groups may exist at the same
        dimension, as for triangular and quadrilateral prism faces.

        At dimension zero the groups cover every node in the complete parent
        layout, not only the topological vertices.  Use :meth:`local_vertices`
        when only the vertex skeleton is required.

        Parameters:
            top_dim: Target dimension in the closed interval
                ``[0, self.top_dim]``.

        Returns:
            Immutable homogeneous local-subentity groups.

        Raises:
            TypeError: If ``top_dim`` is not a plain integer.
            ValueError: If ``top_dim`` is outside the supported interval.
        """
        raise NotImplementedError()

    def vertex_permutations(self) -> tuple[tuple[int, ...], ...]:
        """Return vertex automorphisms supported by this concrete Schema.

        These are reference-entity symmetries that preserve the parameterized
        node set.  They are not an assertion that every vertex ordering is a
        valid orientation.
        """
        raise NotImplementedError()

    def node_permutation(
        self,
        vertex_permutation: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Lift one supported vertex automorphism to all local-node columns.

        ``result[i]`` is the column occupied by local node ``i`` after the
        requested vertex permutation.  The result is a bijection of
        ``range(number_of_nodes())`` and uses the same column convention as
        the geometry and positive-order Lagrange basis APIs.

        Raises:
            TypeError: If the permutation is not a tuple of plain integers.
            ValueError: If it is not supported by this concrete Schema.
        """
        raise NotImplementedError()

    ### [Entity Topology] ###

    @classmethod
    def local_entity(
        cls,
        tgt_name: str,
        /,
        indexing: Literal["o", "s"] = "o",
    ) -> tuple[tuple[int, ...], ...]:
        """Local entity indices of the target entity.

        Parameters:
            tgt_name (str): The name of the target entity.
            indexing (Literal["o", "s"], optional): The indexing method. Defaults to "o".

        Returns:
            tuple[tuple[int, ...], ...]: The immutable local entity indices.
        """
        raise NotImplementedError()

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        raise NotImplementedError()

    ### [Multi-Indices] ###

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """Multi-index of the entity, with one column per vertex."""
        raise NotImplementedError()

    @classmethod
    def multi_index_vertex_columns(cls) -> tuple[int, ...] | None:
        """Map ``multi_index`` columns onto local vertex numbers.

        ``multi_index(..., tensorprod=True)`` emits one column per vertex, but
        a tensor-product Schema numbers those columns in factor-nesting order,
        which need not agree with the vertex numbering used by
        ``local_vertices()``, ``_node_keys()`` and the local entity tables.
        Return the permutation ``columns`` such that column ``s`` of
        ``multi_index`` is the barycentric weight of vertex ``columns[s]``, or
        ``None`` when the two conventions coincide.
        """
        return None

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """Number of multi-indices."""
        return int(cls.multi_index(order, internal=internal).shape[0])

    ### [Geometric Computations] ###

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the barycenter of the entity."""
        raise NotImplementedError()

    @classmethod
    def barycentric[**P, R](
        cls,
        ctx: EntityContext,
        func: Callable[Concatenate[Tensor, P], R],
        index: Index | None
    ) -> Callable[Concatenate[Tensor | tuple[Tensor, ...], P], R]:
        """Transform functions from cartesian to barycentric coordinates."""
        raise NotImplementedError()

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Convert barycentric coordinates to physical points."""
        raise NotImplementedError()

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """Geometric dimension of the cell."""
        raise NotImplementedError()

    def grad_shape_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """Differentiate the geometry basis by barycentric variables.

        The result has shape ``(Q, Lg, B)``.  ``Lg`` follows the complete
        geometry local-node order and ``B`` concatenates barycentric
        components in public reference-factor order.  The barycentric
        variables of each factor are treated as independent.

        Raises:
            NotImplementedError: If the Schema has no independent-
                barycentric gradient contract.
        """
        raise NotImplementedError()

    def grad_shape_function_cartesian(
        self,
        ctx: EntityContext,
        bcs: Tensor | tuple[Tensor, ...],
        *,
        index: Index | None = None
    ) -> Tensor:
        """Differentiate the geometry basis in physical coordinates.

        The geometry basis and all geometry nodes determine the reference-to-
        physical map.  The result's local-basis axis follows the complete
        local-node order.

        Raises:
            NotImplementedError: If the Schema or current embedding does not
                provide this physical-coordinate operation.
        """
        raise NotImplementedError()

    def grad_shape_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """Differentiate the geometry basis in reference coordinates.

        The result has shape ``(Q, Lg, R)`` where ``Lg`` is
        :meth:`number_of_nodes` and ``R`` is the reference dimension.  For
        tensor-product entities, reference components follow the public
        factor order documented by the concrete Schema.
        """
        raise NotImplementedError()

    def lagrange_basis_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """Evaluate an independent Lagrange basis on this reference family.

        This is a reference-entity capability for finite-element spaces; it
        does not change the Schema's geometry order, descriptor, identity, or
        local-node count.  Scalar simplex families require a non-negative
        integer ``p``.  Tensor-product families accept either one
        non-negative integer for every factor or a tuple in public factor
        order.

        ``bcs`` is one ``(Q, V)`` barycentric tensor for a simplex, or a tuple
        containing one tensor per tensor-product factor.  Tensor-product
        points are evaluated as a Cartesian product.  The result has shape
        ``(Q, Lp)``; for positive order its basis axis follows the complete
        local-node order of the equal family member with geometry order
        ``p``.  The output preserves the input floating dtype and device.

        Raises:
            TypeError: If the coordinate container or order type is invalid.
            ValueError: If coordinate shapes or order values are invalid.
            NotImplementedError: If this Schema family does not provide an
                arbitrary-order Lagrange reference basis.
        """
        raise NotImplementedError()

    def grad_lagrange_basis_function_barycentric(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """Differentiate an order-``p`` basis by barycentric variables.

        The result has shape ``(Q, Lp, B)``.  The basis axis uses the same
        ordering as :meth:`lagrange_basis_function`; barycentric components
        are concatenated in public reference-factor order and are treated as
        independent variables.
        """
        raise NotImplementedError()

    def grad_lagrange_basis_function_reference(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...],
    ) -> Tensor:
        """Differentiate an order-``p`` basis in reference coordinates.

        The result has shape ``(Q, Lp, R)``.  The basis axis uses the same
        ordering as :meth:`lagrange_basis_function`, and the reference-
        coordinate axis follows public factor order.
        """
        raise NotImplementedError()

    @classmethod
    def integral(
        cls,
        ctx: EntityContext,
        func: Callable[[Tensor], Tensor] | Callable[[tuple[Tensor, ...]], Tensor],
        q: int, index: Index | None
    ) -> Tensor:
        """Integral of a barycentric function."""
        raise NotImplementedError()

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Jacobi matrix of the transformation from reference to physical element.

        Parameters:
            ctx (EntityContext): The entity context containing the mesh block and sector information.
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            index (Index | None): The index of the entity in the sector, or None for all entities.

        Returns:
            Tensor: The Jacobi matrix with shape (NC, NQ, GD, ref_dim), where
                NC is the number of cells, NQ is the number of points,
                GD is the geometric dimension, and ref_dim is the number of
                reference coordinates (typically equal to the topological
                dimension of the entity).
        """
        raise NotImplementedError()

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the measure of the entity."""
        raise NotImplementedError()

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the normal vector of the entity."""
        raise NotImplementedError()

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device = None) -> "Quadrature":
        """Quadrature formula for the entity."""
        raise NotImplementedError()

    def shape_function(self, bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
        """Evaluate this concrete Schema's geometry interpolation basis.

        Unlike :meth:`lagrange_basis_function`, this method has no order
        argument: the geometry order is fixed by the immutable Schema value.
        The result has shape ``(Q, Lg)`` where ``Lg`` is
        :meth:`number_of_nodes`; its final axis follows the complete local-node
        layout used by sector connectivity.  The output preserves the input
        floating dtype and device.
        """
        raise NotImplementedError()

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the tangent vector of the entity."""
        raise NotImplementedError()


def _freeze_local_entities(
    entities: Mapping[str, Iterable[Iterable[int]]],
) -> Mapping[str, tuple[tuple[int, ...], ...]]:
    """Create a read-only, deeply immutable local-entity mapping."""
    return MappingProxyType(
        {
            name: tuple(tuple(local) for local in local_entities)
            for name, local_entities in entities.items()
        }
    )
