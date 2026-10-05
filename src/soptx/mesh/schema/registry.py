# 移植自 brighthe/fealpy ``fealpy/mesh/schema/registry.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Schema 类型登记表与解析器, 以及实体类型名到拓扑维数的换算工具."""

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
    """把带版本的 Schema 类型 ID 映射到其 Python Schema 类.

    登记表校验描述符的参数名及其规范值; 不可变实例的构造由 ``SchemaResolver`` 负责.
    """

    def __init__(self) -> None:
        self._schema_types: dict[tuple[str, int], type[EntitySchema]] = {}
        self._keys_by_type: dict[type[EntitySchema], tuple[str, int]] = {}
        self._parameter_names: dict[tuple[str, int], tuple[str, ...]] = {}

    @property
    def schema_types(
        self,
    ) -> Mapping[tuple[str, int], type[EntitySchema]]:
        """已登记的类型/版本对的只读视图."""
        return MappingProxyType(self._schema_types)

    def register(self, schema_type: type[EntitySchema]) -> None:
        """按描述符元数据登记一个 Schema Python 类型.

        Parameters
        ----------
        schema_type : type
            声明了 ``type_id``、``schema_version`` 与 ``descriptor_parameter_names`` 的
            ``EntitySchema`` 子类.

        Raises
        ------
        TypeError
            类或其元数据的类型不合法.
        ValueError
            元数据不合法, 或与已有登记冲突.
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
        """返回为某个类型/版本对精确登记的类.

        Raises
        ------
        KeyError
            类型 ID 或所求版本未知.
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
        """按登记的类型约定校验描述符.

        参数集合与顺序须与登记完全一致, 参数值须已是不可变 Schema 类型所产生的规范形式.
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
        """校验并规范地编码一个 Schema 描述符."""
        return _encode_schema_descriptor_unchecked(self.validate(descriptor))

    def decode(self, schema_id: str) -> SchemaDescriptor:
        """解码并校验一个已登记的 Schema 标识串."""
        return self.validate(_decode_schema_descriptor_unchecked(schema_id))


class SchemaResolver:
    """经 Schema 类型登记表构造不可变的 Schema 值."""

    __slots__ = ("_type_registry",)

    def __init__(self, type_registry: SchemaTypeRegistry) -> None:
        self._type_registry = type_registry

    def resolve(
        self,
        descriptor_or_id: SchemaDescriptor | str,
    ) -> EntitySchema:
        """把描述符或 ID 解析为相等的不可变 Schema 值.

        每次调用可能返回一个新的相等实例; 调用方不得以 Python 对象身份作为 Schema 身份.
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
    """把负的拓扑维数 (从最高维倒数) 换算为非负值.

    Raises
    ------
    ValueError
        ``top_dim`` 超出 ``[-highest_dim-1, highest_dim]``.
    """
    if top_dim < -highest_dim - 1 or top_dim > highest_dim:
        raise ValueError(f"top dimension {top_dim} is out of range "
                         f"[-{highest_dim + 1}, {highest_dim}]")
    if top_dim < 0:
        top_dim += highest_dim + 1

    return top_dim


def string_to_etype_and_idx(etype_string: str, highest_dim: int) -> tuple[int, int]:
    """解析 ``"类型名"`` 或 ``"类型名:序号"`` 形式的实体选择串, 返回 ``(拓扑维数, 序号)``."""
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
    """把实体类型名 (cell、face、edge、node, 不区分大小写) 换算为拓扑维数.

    Raises
    ------
    ValueError
        类型名不受支持.
    """
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
