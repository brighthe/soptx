# 移植自 brighthe/fealpy ``fealpy/mesh/schema/registry.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from .descriptor import (
    SchemaDescriptor,
    _decode_schema_descriptor_unchecked,
    _encode_schema_descriptor_unchecked,
)
from .entity_schema import EntitySchema
from .polygon import PolygonSchema
from .classic import (
    HexahedronSchema,
    NodeSchema,
    PrismSchema,
    PyramidSchema,
    QuadrilateralSchema,
    EdgeSchema,
    TetrahedronSchema,
    TriangleSchema,
)

__all__ = [
    "SCHEMA_RESOLVER",
    "SCHEMA_TYPE_REGISTRY",
    "SchemaResolver",
    "SchemaTypeRegistry",
]


class SchemaTypeRegistry:
    """Map versioned schema type IDs to their Python schema classes.

    The registry validates descriptor parameter names and their canonical
    values. Immutable instance construction is owned by ``SchemaResolver``.
    """

    def __init__(self) -> None:
        self._schema_types: dict[tuple[str, int], type[EntitySchema]] = {}
        self._keys_by_type: dict[type[EntitySchema], tuple[str, int]] = {}
        self._parameter_names: dict[tuple[str, int], tuple[str, ...]] = {}

    @property
    def schema_types(
        self,
    ) -> Mapping[tuple[str, int], type[EntitySchema]]:
        """Return a read-only view of registered type/version pairs."""
        return MappingProxyType(self._schema_types)

    def register(self, schema_type: type[EntitySchema]) -> None:
        """Register one schema Python type from its descriptor metadata.

        Parameters:
            schema_type: ``EntitySchema`` subclass declaring ``type_id``,
                ``schema_version``, and ``descriptor_parameter_names``.

        Raises:
            TypeError: If the class or its metadata has an invalid type.
            ValueError: If its metadata is invalid or conflicts with an
                existing registration.
        """
        if not isinstance(schema_type, type) or not issubclass(
            schema_type,
            EntitySchema,
        ):
            raise TypeError(
                "schema_type must be an EntitySchema subclass, "
                f"got {schema_type!r}"
            )

        type_id = getattr(schema_type, "type_id", None)
        schema_version = getattr(schema_type, "schema_version", None)
        parameter_names = getattr(
            schema_type,
            "descriptor_parameter_names",
            None,
        )
        if type(parameter_names) is not tuple:
            raise TypeError(
                f"{schema_type.__name__}.descriptor_parameter_names must "
                f"be a tuple, got {type(parameter_names).__name__}"
            )

        metadata = SchemaDescriptor(
            type_id=type_id,
            schema_version=schema_version,
            parameters=tuple((name, 0) for name in parameter_names),
        )
        key = (metadata.type_id, metadata.schema_version)

        if key in self._schema_types:
            registered = self._schema_types[key]
            raise ValueError(
                f"schema type key {key!r} is already registered by "
                f"{registered.__name__}"
            )
        if schema_type in self._keys_by_type:
            registered_key = self._keys_by_type[schema_type]
            raise ValueError(
                f"schema type {schema_type.__name__} is already registered "
                f"as {registered_key!r}"
            )

        self._schema_types[key] = schema_type
        self._keys_by_type[schema_type] = key
        self._parameter_names[key] = parameter_names

    def get(self, type_id: str, schema_version: int) -> type[EntitySchema]:
        """Return the class registered for an exact type/version pair.

        Raises:
            KeyError: If the type ID or its requested version is unknown.
        """
        key = (type_id, schema_version)
        try:
            return self._schema_types[key]
        except KeyError:
            versions = sorted(
                version
                for registered_type, version in self._schema_types
                if registered_type == type_id
            )
            if versions:
                raise KeyError(
                    f"unknown schema version {schema_version} for type_id "
                    f"{type_id!r}; registered versions are {versions}"
                ) from None
            raise KeyError(f"unknown schema type_id {type_id!r}") from None

    def validate(self, descriptor: SchemaDescriptor) -> SchemaDescriptor:
        """Validate a descriptor against its registered type contract.

        The parameter set and order must exactly match the registration.
        Parameter values must already be in the canonical form produced by
        the immutable schema type.
        """
        if not isinstance(descriptor, SchemaDescriptor):
            raise TypeError(
                "descriptor must be a SchemaDescriptor, "
                f"got {type(descriptor).__name__}"
            )

        self.get(descriptor.type_id, descriptor.schema_version)
        key = (descriptor.type_id, descriptor.schema_version)
        expected = self._parameter_names[key]
        actual = tuple(name for name, _ in descriptor.parameters)
        if actual != expected:
            missing = tuple(name for name in expected if name not in actual)
            extra = tuple(name for name in actual if name not in expected)
            details: list[str] = []
            if missing:
                details.append(f"missing parameters {missing!r}")
            if extra:
                details.append(f"unexpected parameters {extra!r}")
            if not details:
                details.append(
                    f"parameter order must be {expected!r}, got {actual!r}"
                )
            raise ValueError(
                f"invalid descriptor for {descriptor.type_id!r}@"
                f"{descriptor.schema_version}: {'; '.join(details)}"
            )

        schema_type = self.get(
            descriptor.type_id,
            descriptor.schema_version,
        )
        schema = schema_type(**dict(descriptor.parameters))
        normalized = schema.descriptor
        if normalized != descriptor:
            expected_id = _encode_schema_descriptor_unchecked(normalized)
            actual_id = _encode_schema_descriptor_unchecked(descriptor)
            raise ValueError(
                "schema descriptor parameters are not canonical; "
                f"expected {expected_id!r}, got {actual_id!r}"
            )
        return descriptor

    def encode(self, descriptor: SchemaDescriptor) -> str:
        """Validate and canonically encode one schema descriptor."""
        return _encode_schema_descriptor_unchecked(self.validate(descriptor))

    def decode(self, schema_id: str) -> SchemaDescriptor:
        """Decode and validate one registered schema identifier."""
        return self.validate(_decode_schema_descriptor_unchecked(schema_id))


class SchemaResolver:
    """Construct immutable Schema values through a Schema type registry."""

    __slots__ = ("_type_registry",)

    def __init__(self, type_registry: SchemaTypeRegistry) -> None:
        self._type_registry = type_registry

    def resolve(
        self,
        descriptor_or_id: SchemaDescriptor | str,
    ) -> EntitySchema:
        """Resolve a descriptor or ID to an equal immutable schema value.

        A new equal instance may be returned on each call. Callers must not
        use Python object identity as schema identity.
        """
        if isinstance(descriptor_or_id, str):
            descriptor = self._type_registry.decode(descriptor_or_id)
        elif isinstance(descriptor_or_id, SchemaDescriptor):
            descriptor = self._type_registry.validate(descriptor_or_id)
        else:
            raise TypeError(
                "descriptor_or_id must be a SchemaDescriptor or string, "
                f"got {type(descriptor_or_id).__name__}"
            )
        schema_type = self._type_registry.get(
            descriptor.type_id,
            descriptor.schema_version,
        )
        return schema_type(**dict(descriptor.parameters))


SCHEMA_TYPE_REGISTRY = SchemaTypeRegistry()
for _schema_type in (
    NodeSchema,
    EdgeSchema,
    TriangleSchema,
    QuadrilateralSchema,
    TetrahedronSchema,
    PrismSchema,
    PyramidSchema,
    HexahedronSchema,
    PolygonSchema,
):
    SCHEMA_TYPE_REGISTRY.register(_schema_type)

SCHEMA_RESOLVER = SchemaResolver(SCHEMA_TYPE_REGISTRY)


def ensure_positive_topdim(top_dim: int, highest_dim: int) -> int:
    if top_dim < -highest_dim - 1 or top_dim > highest_dim:
        raise ValueError(f"top dimension {top_dim} is out of range "
                         f"[-{highest_dim + 1}, {highest_dim}]")
    if top_dim < 0:
        top_dim += highest_dim + 1

    return top_dim


def string_to_etype_and_idx(etype_string: str, highest_dim: int) -> tuple[int, int]:
    etype_idx = etype_string.split(":")

    if len(etype_idx) == 1:
        etype = etype_idx[0].strip()
        idx = 0
    else:
        etype, idx = etype_idx
        idx = int(idx.strip())

    topdim = etype_to_topdim(etype, highest_dim)
    return topdim, idx


def etype_to_topdim(etype: str, highest_dim: int) -> int:
    etype = etype.upper()

    if etype == "CELL":
        return highest_dim
    if etype == "FACE":
        return highest_dim - 1
    if etype == "EDGE":
        return 1
    if etype == "NODE":
        return 0

    raise ValueError(f"etype name {etype} is not supported, "
					 "available options are: cell, face, edge, node.")
