# 移植自 brighthe/fealpy ``fealpy/mesh/view/entity_view.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Sector-bound homogeneous entity views.

``EntityView`` exposes geometry, quadrature, interpolation and relation access
for one ``EntitySector`` inside a :class:`MeshBlock`.  The module also keeps
private ``_legacy_*`` basis-order adapters used by the classic ``MeshView``
historical shape-function contract; those helpers are not a public basis API
and new FunctionSpace code should call ``EntitySchema`` or ``EntityView``
interfaces directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, final, Literal, ParamSpec, TYPE_CHECKING

from ...backend import bm, Tensor, Index
from ..schema.entity_schema import EntitySchema
from ..schema.classic.base import _TensorProductOrderSchema
from ..storage import EntityContext

if TYPE_CHECKING:
    from ..storage import MeshBlock, EntitySector, Relation
    from ..topology.boundary import BoundaryInfo

__all__ = ["EntityView"]

P = ParamSpec("P")


def _normalized_legacy_order(
    schema: EntitySchema,
    p: int | tuple[int, ...],
) -> int | tuple[int, ...]:
    """Normalize an order for the private classic-basis column adapter."""
    factor_count = getattr(schema, "factor_count", None)
    if factor_count is None:
        if type(p) is int:
            return p
        if type(p) is tuple and len(p) == 1:
            return p[0]
        return p
    if type(p) is int:
        return (p,) * factor_count
    return p


def _legacy_basis_indices(
    schema: EntitySchema,
    p: int | tuple[int, ...],
) -> Tensor | None:
    """Map the new local-node basis order back to the classic View order."""
    permutation_getter = getattr(schema, "_lagrange_basis_permutation", None)
    if permutation_getter is None:
        return None

    order = _normalized_legacy_order(schema, p)
    permutation = permutation_getter(order)
    if permutation is None:
        return None

    # 张量积单元 (四边形/六面体/三棱柱) 不做这次反置换.
    #
    # lagrange_basis_function 已经把基函数排成 _node_keys() 序, 而
    # cell_to_ipoint 给出的局部自由度也正是这个序; 再套一次逆置换会把列序
    # 打回张量积枚举序 (四边形 p=1 即 (0,0),(1,0),(0,1),(1,1)), 与自由度
    # 错位. 后果是静默的: 残差照样收敛, 只有观测收敛阶会塌掉 (Q1 由 2.0
    # 掉到 0.11).
    #
    # 原判据只豁免 prism / hexahedron 且要求各方向阶次全为 1, 于是
    # 六面体 p=1 侥幸正确, 四边形所有阶次以及 p >= 2 的六面体/三棱柱全错.
    # 单纯形的 _lagrange_basis_permutation 是恒等置换, 反置换对它无害,
    # 因此这里只需按张量积与否豁免, 不再看阶次.
    if isinstance(schema, _TensorProductOrderSchema):
        return None

    inverse = [0] * len(permutation)
    for local_index, kernel_index in enumerate(permutation):
        inverse[kernel_index] = local_index
    return bm.asarray(inverse, dtype=bm.int64)


def _restore_legacy_basis_order(
    values: Tensor,
    schema: EntitySchema,
    p: int | tuple[int, ...],
    *,
    gradient: bool,
) -> Tensor:
    """Apply the private FEALPy-style View basis-column convention."""
    indices = _legacy_basis_indices(schema, p)
    if indices is None:
        return values
    indices = bm.device_put(indices, bm.get_device(values))
    if gradient:
        return values[..., indices, :]
    return values[..., indices]


def _pyramid_geometry_order(p: int | tuple[int, ...]) -> bool:
    """Return whether a transitional Pyramid View request is geometry p=1."""
    return p == 1 or p == (1,)


@final
@dataclass(slots=True)
class EntityView:
    """View of one homogeneous mesh entity sector.

    ``EntityView`` binds a mesh block to one sector by its stable sector id.
    It does not own a second copy of :class:`EntitySector` state; :attr:`sector`
    and :attr:`schema` resolve against the mesh block on every access.

    Attributes:
        block: Mesh storage block containing coordinates, sectors, and
            relations.
        sector_id: Stable sector id resolved through ``block.sectors``.
    """
    block: MeshBlock
    sector_id: str

    def __post_init__(self) -> None:
        """Validate the bound sector id after dataclass initialization."""
        if type(self.sector_id) is not str or not self.sector_id:
            raise TypeError("EntityView.sector_id must be a non-empty string")
        self.block.get_sector(self.sector_id)

    @property
    def sector(self) -> EntitySector:
        """Return the bound :class:`EntitySector` from the mesh block."""
        return self.block.get_sector(self.sector_id)

    @property
    def schema(self) -> EntitySchema:
        """Return the concrete immutable Schema value bound by the sector."""
        return self.sector.schema

    def __len__(self) -> int:
        """Return the number of entities in this sector."""
        return self.schema.size(self.context())

    def context(self) -> EntityContext:
        """Return the schema context for this entity view.

        Returns:
            EntityContext: Lightweight container holding the mesh block and the
            current entity sector.
        """
        return EntityContext(self.block, self.sector)

    def _selected_connectivity(self, index: Index | None) -> Tensor:
        """Return selected sector connectivity with an explicit entity axis."""
        connectivity = self.sector.indices
        if index is None:
            return connectivity
        selected = connectivity[index]
        if len(selected.shape) == 1:
            selected = bm.reshape(selected, (1, -1))
        return selected

    def _default_geometry_quadrature_order(self) -> int:
        """Return the deterministic default geometry quadrature order."""
        order = getattr(self.schema, "p", None)
        if order is None:
            return 1
        if isinstance(order, tuple):
            order = max(order)
        return max(int(order), 1) + 1

    def _reference_center_bcs(self) -> tuple[Tensor, ...]:
        """Return the reference-entity canonical center barycentric input."""
        if self.schema.type_id == "lagrange_pyramid":
            center = bm.asarray([[0.5, 0.5]], dtype=bm.float64)
            return (center, center, center)
        quadrature = self.schema.quadrature_formula(1)
        bcs, _ = quadrature.get_quadrature_points_and_weights()
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        return bcs

    # User APIs

    def barycentric[**P, R](self, func: Callable[Concatenate[Tensor, P], R], /, *, index: Index | None = None):
        """Wrap a Cartesian-coordinate function as a barycentric function.

        Parameters:
            func (Callable): Function whose first positional argument is a
                physical coordinate tensor.  The expected coordinate shape is
                determined by the entity schema.
            index (Index, optional): Entity subset used when converting
                barycentric coordinates to physical points.

        Returns:
            Callable: Function with the same remaining arguments as ``func``
            whose first positional argument is a barycentric coordinate tensor,
            or a tuple of barycentric coordinate tensors for tensor-product
            entities.
        """
        return self.schema.barycentric(self.context(), func, index)

    def barycenter(self, *, index: Index | None = None) -> Tensor:
        """Return the mapped reference center of selected entities.

        The reference center is mapped through the same all-node geometry
        interpolation used by :meth:`bc_to_point`.  For affine entities this
        equals the usual vertex average; for curved entities it is not the
        measure-weighted centroid.

        Parameters:
            index (Index, optional): Entity subset.  If ``None``, all entities
                in this sector are used.

        Returns:
            Tensor: Barycenter coordinates with shape ``(NE, GD)`` or the
            indexed subset shape, where ``GD`` is the geometric dimension.
        """
        if self.sector.indptr is not None:
            return self.schema.barycenter(self.context(), index)
        bcs = self._reference_center_bcs()
        points = self.bc_to_point(bcs, index=index)
        return bm.reshape(points, (points.shape[0], points.shape[-1]))

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """Map barycentric coordinates to physical coordinates.

        The mapping uses the complete geometry local-node layout, so curved
        high-order entities are evaluated with all geometry nodes rather than
        only their vertex skeleton.

        Parameters:
            bc (Tensor | tuple[Tensor, ...]): Barycentric coordinates.  Simplex
                entities use a single tensor, while tensor-product entities may
                use one tensor per factor.
            index (Index, optional): Entity subset on which the mapping is
                evaluated.

        Returns:
            Tensor: Physical points.  For common simplex entities the shape is
            ``(NE, NQ, GD)`` after selecting entities and quadrature points.

        Raises:
            TypeError: If the coordinate container type is invalid.
            ValueError: If barycentric shapes are inconsistent with the bound
                Schema.
        """
        if not isinstance(bc, tuple):
            bc = (bc,)
        values = self.schema.shape_function(bc)
        points = self.block.positions[self._selected_connectivity(index)]
        return bm.einsum("qi,cij->cqj", values, points)

    def boundary(self) -> "BoundaryInfo":
        """Infer boundary information for this entity sector.

        Returns:
            BoundaryInfo:
            - mask: Boolean tensor indicating which entities are on the boundary.
            - index: Integer tensor of boundary entity indices.
        """
        from ..topology.boundary import BoundaryInferencer

        if self.block._cache_boundary_info is None:
            self.block._cache_boundary_info = {}
        return BoundaryInferencer.infer_entity(
            self.block,
            self.sector_id,
            self.block._cache_boundary_info,
        )

    def del_attribute(self, name: str) -> None:
        """Delete a user attribute from the entity sector.

        Parameters:
            name (str): Attribute name.

        Raises:
            KeyError: If ``name`` is not present in ``sector.attributes``.
        """
        if name in self.sector.attributes:
            del self.sector.attributes[name]
        else:
            raise KeyError(f"Attribute '{name}' not found in sector attributes.")

    def error(
        self,
        f1: Callable[..., Tensor],
        f2: Callable[..., Tensor],
        /,
        power: float = 2.0,
        q: int = 3,
        *,
        cell_axis: bool = False,
        index: Index | None = None
    ) -> Tensor:
        """Compute an integral error norm between two functions on the entity.

        Functions not marked as barycentric are wrapped with
        :meth:`barycentric` before integration.  The computed value is
        ``(integral(abs(f1 - f2)**power))**(1/power)``.

        Parameters:
            f1 (Callable[..., Tensor]): First function.
            f2 (Callable[..., Tensor]): Second function.
            power (float, optional): Norm power.  Default is 2.0.
            q (int, optional): Quadrature order.  Default is 3.
            cell_axis (bool, optional): If ``True``, return one error value per
                selected entity.  If ``False``, return the global error over
                all selected entities.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Scalar global error, or a tensor of entity-wise errors when
            ``cell_axis`` is ``True``.
        """
        from ...decorator import barycentric
        if not getattr(f1, "coordtype", None) == "barycentric":
            f1 = self.barycentric(f1, index=index)
        if not getattr(f2, "coordtype", None) == "barycentric":
            f2 = self.barycentric(f2, index=index)
        @barycentric
        def integrand(bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
            v1 = f1(bcs)
            v2 = f2(bcs)
            return bm.abs(v1 - v2) ** power

        if cell_axis:
            return self.integral(integrand, q=q, index=index) ** (1.0 / power)
        return bm.sum(self.integral(integrand, q=q, index=index)) ** (1.0 / power)

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the embedding space."""
        return self.schema.geo_dimension(self.context())

    def get_attribute(self, name: str) -> Any:
        """Return a user attribute stored on the entity sector.

        Parameters:
            name (str): Attribute name.

        Returns:
            Any: Stored attribute value, or ``None`` if the attribute does not
            exist.
        """
        return self.sector.attributes.get(name)

    def global_permutations(
        self,
        name_or_topdim: str | int | EntityView,
        idx: int = 0,
        indexing: Literal["o", "s"] = "o",
    ) -> Tensor:
        """Return local-to-global orientation permutations for sub-entities.

        Parameters:
            name_or_topdim (str | int | EntityView): Target sub-entity schema
                name, topological dimension, or explicit ``EntityView``.
            idx (int, optional): Target sector index when ``name_or_topdim``
                selects a dimension or entity type that has multiple sectors.  Default is 0.

        Returns:
            Tensor: Integer tensor whose leading axes enumerate source entities
            and their local target entities.  The last axis stores the vertex
            permutation induced by global orientation.
        """
        if indexing != "o":
            raise NotImplementedError(
                "only indexing='o' is supported by explicit-sector orientation"
            )

        if not isinstance(name_or_topdim, EntityView):
            raise TypeError("global_permutations expects an EntityView")
        tgt = name_or_topdim

        from ...mesh.schema.utils import argpermute
        from ...mesh.topology.relation import resolve_relation

        groups = self.schema.local_entity_groups(tgt.top_dimension())
        candidates = [group for group in groups if group.schema == tgt.schema]
        if not candidates:
            raise ValueError(
                f"target sector {tgt.sector_id!r} is not a local subentity "
                f"of {self.sector_id!r}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous local subentity groups for target "
                f"{tgt.sector_id!r}"
            )

        child_vertices = tgt.schema.local_vertices()
        local_face_vertices = [
            tuple(row[vertex] for vertex in child_vertices)
            for row in candidates[0].local_node_indices
        ]
        local_face = bm.asarray(local_face_vertices, dtype=bm.int64)
        parent_vertices = self.indices[:, local_face]

        target_indices = tgt.indices[:, child_vertices]
        cell_to_target = resolve_relation(
            self.block,
            self.sector_id,
            tgt.sector_id,
        ).tgt_indices

        return argpermute(
            parent_vertices,
            target_indices[cell_to_target],
            dtype=bm.uint8,
        )

    def grad_lambda(
        self,
        bcs: tuple[Tensor, ...] | None = None,
        index: Index | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """Return gradients of barycentric coordinate functions.

        Parameters:
            bcs (tuple[Tensor, ...] | None, optional): Evaluation points in
                barycentric coordinates.  If provided, gradients may be
                broadcast along the quadrature-point axis.
            index (Index, optional): Entity subset.
            ref (bool, optional): If ``True``, return gradients on the reference
                entity.  If ``False``, return gradients in physical coordinates.
                Default is ``False``.

        Returns:
            Tensor: Gradients of barycentric coordinates.  Without ``bcs``, a
            typical shape is ``(NE, NV, GD)`` for physical gradients or
            ``(NE, NV, NR)`` for reference gradients.
        """
        return self.schema.grad_lambda(self.context(), index, bcs=bcs, ref=ref) # type: ignore

    def grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """Evaluate gradients of this sector's geometry shape functions.

        The geometry order is fixed by the bound Schema value; this method does
        not accept an independent solution order.  The result has shape
        ``(Q, Lg, R)`` where ``Lg`` is the Schema's complete local-node count
        and ``R`` is its reference dimension.
        """
        return self.schema.grad_shape_function_reference(bcs)

    def grad_shape_function_cartesian(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        *,
        index: Index | None = None,
    ) -> Tensor:
        """Evaluate geometry shape-function gradients in physical coordinates.

        The result has shape ``(NE, Q, Lg, GD)``.  It is obtained by mapping
        the reference gradient through the current all-node Jacobian and the
        reference metric, so curved geometry is not silently replaced by a
        vertex-only affine formula.

        Parameters:
            bcs: Barycentric evaluation points, either one tensor for simplex
                entities or one tensor per tensor-product factor.
            index: Optional entity subset.

        Returns:
            Physical-coordinate gradients with shape ``(NE, Q, Lg, GD)``.
        """
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        reference_gradient = self.schema.grad_shape_function_reference(bcs)
        jacobian = self.jacobi_matrix(bcs, index=index)
        metric = bm.einsum("cqdr,cqds->cqrs", jacobian, jacobian)
        metric_inverse = bm.linalg.inv(metric)
        return bm.einsum(
            "cqdr,cqrs,qis->cqid",
            jacobian,
            metric_inverse,
            reference_gradient,
        )

    def _legacy_grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: Literal["b", "u", "x"] = "u",
        mi = None,
    ) -> Tensor:
        """Private classic arbitrary-order gradient entry point.

        This helper preserves the historical FEALPy ``grad_shape_function(p)``
        column convention for classic ``MeshView`` methods.  New code should
        use :meth:`EntityView.grad_shape_function` or
        ``EntitySchema.grad_lagrange_basis_function_reference`` directly.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if variables == "b":
            if (
                self.schema.type_id == "lagrange_pyramid"
                and _pyramid_geometry_order(p)
            ):
                return self.schema.grad_shape_function_barycentric(bcs)
            grad = self.schema.grad_lagrange_basis_function_barycentric(
                bcs,
                p,
            )
            return _restore_legacy_basis_order(
                grad,
                self.schema,
                p,
                gradient=True,
            )
        elif variables == "u":
            if (
                self.schema.type_id == "lagrange_pyramid"
                and _pyramid_geometry_order(p)
            ):
                grad = self.schema.grad_shape_function_reference(bcs)
            else:
                grad = self.schema.grad_lagrange_basis_function_reference(
                    bcs,
                    p,
                )
            return _restore_legacy_basis_order(
                grad,
                self.schema,
                p,
                gradient=True,
            )
        elif variables == "x":
            from ..mapping import piola_transform_covariant

            if (
                self.schema.type_id == "lagrange_pyramid"
                and _pyramid_geometry_order(p)
            ):
                grad_ref = self.schema.grad_shape_function_reference(bcs)
            else:
                grad_ref = self.schema.grad_lagrange_basis_function_reference(
                    bcs,
                    p,
                )
            grad_ref = _restore_legacy_basis_order(
                grad_ref,
                self.schema,
                p,
                gradient=True,
            )[None, ...]
            J = self.schema.jacobi_matrix(
                self.context(),
                bcs,
                index,
            )[..., None, :, :]
            return piola_transform_covariant(grad_ref, J)
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    @property
    def indices(self) -> Tensor:
        """Connectivity array of this entity sector."""
        return getattr(self.sector, "indices")

    @property
    def indptr(self) -> Tensor | None:
        """Return ragged connectivity offsets, or ``None`` when homogeneous."""
        return self.sector.indptr

    def integral(
        self,
        func: Callable[..., Tensor],
        /,
        q: int = 3,
        *,
        index: Index | None = None
    ) -> Tensor:
        """Integrate a barycentric function over selected entities.

        Parameters:
            func (Callable[..., Tensor]): Function evaluated at barycentric
                quadrature points.
            q (int, optional): Quadrature order.  Default is 3.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Integral values.  For scalar integrands this is typically
            one value per selected entity before any caller-side reduction.
        """
        return self.schema.integral(self.context(), func, q, index)

    def jacobi_matrix(self, bcs: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """Return Jacobian matrices of the reference-to-physical map.

        The Jacobian is computed from the geometry reference gradient and all
        geometry nodes:

        ``J[c, q, d, r] = sum_i X[c, i, d] * dphi_ref[q, i, r]``

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric evaluation points.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Jacobian tensor, commonly with shape ``(NE, NQ, GD, ref_dim)``,
            where ``ref_dim`` is the reference dimension.
        """
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        gradients = self.schema.grad_shape_function_reference(bcs)
        points = self.block.positions[self._selected_connectivity(index)]
        return bm.einsum("cij,qir->cqjr", points, gradients)

    def metric_density(self, bcs: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        """Return the unsigned metric density of the reference-to-physical map.

        For reference dimension ``r > 0`` this is ``sqrt(det(J^T J))``.  For a
        zero-dimensional reference entity it follows the schema's zero-
        dimensional convention and returns one density per sample.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...]): Barycentric evaluation points.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Metric density with shape ``(NE, NQ)``.
        """
        jacobian = self.jacobi_matrix(bcs, index=index)
        ref_dim = int(jacobian.shape[-1])
        if ref_dim == 0:
            return bm.ones(jacobian.shape[:2], dtype=jacobian.dtype, device=bm.get_device(jacobian))
        metric = bm.einsum("cqdr,cqds->cqrs", jacobian, jacobian)
        return bm.sqrt(bm.linalg.det(metric))

    def measure(self, *, index: Index | None = None, q: int | None = None) -> Tensor:
        """Return the physical measure of selected entities.

        The measure is obtained by integrating the unsigned metric density over
        the reference entity.  When ``q`` is omitted, the Schema's deterministic
        default geometry quadrature order is used; callers may raise ``q`` to
        check convergence for curved geometry.

        Parameters:
            index (Index, optional): Entity subset.
            q (int, optional): Geometry quadrature order.  If ``None``, the
                default policy is used.

        Returns:
            Tensor: One measure per selected entity.  The measure is length for
            edges, area for surface entities, and volume for volume entities.
        """
        if self.sector.indptr is not None:
            return self.schema.measure(self.context(), index)  # type: ignore[attr-defined]
        if q is None:
            q = self._default_geometry_quadrature_order()
        quadrature = self.schema.quadrature_formula(q)
        bcs, weights = quadrature.get_quadrature_points_and_weights()
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        density = self.metric_density(bcs, index=index)
        measure = bm.einsum("cq,q->c", density, weights)
        ref_measure = getattr(self.schema, "ref_measure", None)
        if ref_measure is not None:
            measure = measure * ref_measure
        return measure

    def multi_index_matrix(self, order: int | tuple[int, ...], *, internal: bool = False, tensorprod: bool = True):
        """Return interpolation multi-indices for this entity type.

        Parameters:
            order (int | tuple[int, ...]): Polynomial degree, or tensor-product
                degrees.
            internal (bool, optional): If ``True``, return only interior
                interpolation-point multi-indices.  Default is ``False``.
            tensorprod (bool, optional): If ``True``, use the tensor-product
                ordering expected by interpolation utilities.  Default is
                ``True``.

        Returns:
            Tensor: Integer tensor with one row per local interpolation point
            and one column per local vertex or tensor-product coordinate.
        """
        if isinstance(order, int):
            order = (order,)
        return self.schema.multi_index(order, internal=internal, tensorprod=tensorprod)

    def normal(self, bcs: Tensor | tuple[Tensor, ...] | None = None, *, index: Index | None = None) -> Tensor:
        """Return intrinsic normal vectors associated with selected entities.

        When ``bcs`` is provided, the normal is evaluated pointwise from the
        all-node Jacobian.  Without ``bcs`` the historical convenience behavior
        is preserved for affine sectors.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...] | None, optional): Barycentric
                evaluation points.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Normal vectors.  Schemas commonly return shape
            ``(NE, NN, GD)`` without ``bcs`` and ``(NE, Q, NN, GD)`` with
            ``bcs``, where ``NN`` is the number of normal directions.

        Raises:
            NotImplementedError: If a pointwise normal frame is requested for a
                higher-codimension entity whose convention is not frozen.
        """
        if bcs is None:
            return self.schema.normal(self.context(), index)
        if not isinstance(bcs, tuple):
            bcs = (bcs,)

        tangent = self.tangent(bcs, index=index)
        gd = self.geo_dimension()
        td = self.top_dimension()
        if gd == td:
            return bm.zeros(
                tangent.shape[:2] + (0, gd),
                dtype=tangent.dtype,
                device=bm.get_device(tangent),
            )
        if gd == td + 1:
            if td == 1:
                edge = tangent[..., 0, :]
                normal = bm.stack([edge[..., 1], -edge[..., 0]], axis=-1)
                return bm.expand_dims(normal, axis=2)
            if td == 2:
                normal = bm.cross(
                    tangent[..., 0, :],
                    tangent[..., 1, :],
                    axis=-1,
                )
                return bm.expand_dims(normal, axis=2)
        raise NotImplementedError(
            f"pointwise normal frame is not frozen for codimension "
            f"{gd - td} sector {self.sector_id!r}"
        )

    def num_multi_index(self, order: int | tuple[int, ...], *, internal: bool = False) -> int:
        """Return the number of local interpolation multi-indices.

        Parameters:
            order (int | tuple[int, ...]): Polynomial degree, or tensor-product
                degrees.
            internal (bool, optional): If ``True``, count only interior
                interpolation points.  Default is ``False``.

        Returns:
            int: Number of local multi-indices for the requested order.
        """
        if isinstance(order, int):
            order = (order,)
        return self.schema.num_multi_index(order, internal=internal)

    def quadrature_formula(self, q: int = 3, qtype: str = "legendre"):
        """Return a quadrature formula on the reference entity.

        Parameters:
            q (int, optional): Quadrature order.  Default is 3.
            qtype (str, optional): Quadrature family.  Default is
                ``"legendre"``.

        Returns:
            Quadrature: Quadrature object supplied by the entity schema.
        """
        return self.schema.quadrature_formula(q, qtype)

    def set_attribute(self, name: str, value: Any) -> None:
        """Set a user attribute on the entity sector.

        Parameters:
            name (str): Attribute name.
            value (Any): Attribute value.
        """
        self.sector.attributes[name] = value

    def shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
    ) -> Tensor:
        """Evaluate this sector's geometry shape functions.

        The output follows the concrete Schema's complete local-node layout,
        so its final axis equals both ``schema.number_of_nodes()`` and the
        connectivity width of ``indices``.  It does not accept an independent
        solution order.
        """
        return self.schema.shape_function(bcs)

    def _legacy_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi = None,
    ) -> Tensor:
        """Private classic arbitrary-order basis entry point.

        This helper preserves the historical FEALPy ``shape_function(p)``
        column convention for classic ``MeshView`` methods.  New code should
        use :meth:`EntityView.shape_function` or
        ``EntitySchema.lagrange_basis_function`` directly.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if (
            self.schema.type_id == "lagrange_pyramid"
            and _pyramid_geometry_order(p)
        ):
            val = self.schema.shape_function(bcs)
        else:
            val = self.schema.lagrange_basis_function(bcs, p)
        val = _restore_legacy_basis_order(
            val,
            self.schema,
            p,
            gradient=False,
        )
        if variables == "u":
            return val
        elif variables == "x":
            return val[None, ...] # type: ignore[return-value]
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    def size(self) -> int:
        """Return the number of entities in this sector."""
        return self.schema.size(self.context())

    def tangent(self, bcs: Tensor | tuple[Tensor, ...] | None = None, *, index: Index | None = None) -> Tensor:
        """Return tangent vectors associated with selected entities.

        When ``bcs`` is provided, the tangent frame is the pointwise all-node
        Jacobian with the reference axis transposed last.  Without ``bcs`` the
        historical convenience behavior is preserved for affine sectors.

        Parameters:
            bcs (Tensor | tuple[Tensor, ...] | None, optional): Barycentric
                evaluation points.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Tangent vectors, commonly with shape ``(NE, TD, GD)``
            without ``bcs`` and ``(NE, Q, TD, GD)`` with ``bcs``.
        """
        if bcs is None:
            return self.schema.tangent(self.context(), index)
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        jacobian = self.jacobi_matrix(bcs, index=index)
        return bm.swapaxes(jacobian, -1, -2)

    def to(self, target: int | str | EntityView, idx: int = 0, /) -> Relation:
        """Return the relation from this entity sector to a target sector.

        Parameters:
            target (int | str | EntityView): Target entity selector.  It may be
                a topological dimension, an entity type/name, or another
                ``EntityView``.
            idx (int, optional): Target sector index when ``target`` selects a
                dimension or entity type that has multiple sectors.  Default is
                0.

        Returns:
            Relation: Relation object containing source and target indices and
            conversion helpers such as array or COO representations.
        """
        from ..topology.relation import resolve_relation

        if isinstance(target, EntityView):
            tgt = target.sector_id
        elif isinstance(target, str) and target in self.block.sectors:
            tgt = target
        elif isinstance(target, int):
            top_dim = target + self.top_dimension() + 1 if target < 0 else target
            names = [
                sector.id
                for sector in self.block.sectors.values()
                if sector.schema.top_dim == top_dim
            ]
            if idx >= len(names):
                raise IndexError(f"no target sector at index {idx} for top_dim {top_dim}")
            tgt = names[idx]
        else:
            raise TypeError("EntityView.to expects an EntityView, sector id, or int")
        return resolve_relation(self.block, self.sector_id, tgt)

    def to_ipoint(self, order: int, index: Index | None = None) -> Tensor:
        """Map entities to global interpolation-point indices.

        Parameters:
            order (int): Interpolation order.
            index (Index, optional): Entity subset.

        Returns:
            Tensor: Integer tensor of shape ``(NE, NIP)`` or the indexed subset,
            where ``NIP`` is the number of local interpolation points on this
            entity type.
        """
        from ..ipoints import to_ipoint
        from .mesh_view import MeshView
        root_id = self.sector.source_cell_sector_id or self.sector_id
        mapping = to_ipoint(
            MeshView(self.block, cell_sector_id=root_id),
            self,
            order,
        )
        return mapping if index is None else mapping[index]

    def top_dimension(self) -> int:
        """Return the topological dimension of this entity type."""
        return self.schema.top_dim
