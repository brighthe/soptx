# 移植自 brighthe/fealpy ``fealpy/mesh/schema/descriptor.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""参数化实体 Schema 的稳定描述符及其编解码."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "CanonicalValue",
    "SchemaDescriptor",
    "decode_schema_descriptor",
    "encode_schema_descriptor",
]


CanonicalValue: TypeAlias = bool | int | str | tuple["CanonicalValue", ...]

_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)")


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must match '[a-z][a-z0-9_]*', got {value!r}"
        )
    return value


def _validate_canonical_value(value: object, path: str) -> None:
    if type(value) in (bool, int, str):
        return
    if type(value) is tuple:
        for index, item in enumerate(value):
            _validate_canonical_value(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} must contain only bool, int, str, or nested tuple values, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class SchemaDescriptor:
    """描述一个参数化实体 Schema 的稳定身份.

    参数以有序对给出, 因为其顺序是 Schema 类型带版本的序列化约定的一部分. 参数值限于与
    后端无关的不可变标量及嵌套元组.

    Parameters
    ----------
    type_id : str
        稳定的登记与序列化键.
    schema_version : int
        描述符解释方式的版本号, 为正整数.
    parameters : tuple of tuple
        承载身份的参数名与规范化的值, 按登记的 Schema 类型所声明的顺序.

    Raises
    ------
    TypeError
        某字段的 Python 类型不受支持.
    ValueError
        标识符、版本或参数布局不合法.
    """

    type_id: str
    schema_version: int
    parameters: tuple[tuple[str, CanonicalValue], ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.type_id, "type_id")
        if type(self.schema_version) is not int:
            raise TypeError(
                "schema_version must be an integer, "
                f"got {type(self.schema_version).__name__}"
            )
        if self.schema_version < 1:
            raise ValueError(
                f"schema_version must be positive, got {self.schema_version}"
            )
        if type(self.parameters) is not tuple:
            raise TypeError(
                "parameters must be a tuple of (name, value) pairs, "
                f"got {type(self.parameters).__name__}"
            )

        names: set[str] = set()
        for index, parameter in enumerate(self.parameters):
            if type(parameter) is not tuple or len(parameter) != 2:
                raise TypeError(
                    "parameters must contain two-item tuples, "
                    f"got {parameter!r} at index {index}"
                )
            name, value = parameter
            _validate_identifier(name, f"parameters[{index}].name")
            if name in names:
                raise ValueError(f"duplicate schema parameter {name!r}")
            names.add(name)
            _validate_canonical_value(value, f"parameter {name!r}")

    def to_id(self) -> str:
        """把本描述符编码为确定的 Schema 标识串."""
        return encode_schema_descriptor(self)

    @classmethod
    def from_id(cls, schema_id: str) -> "SchemaDescriptor":
        """用进程级的类型登记表解码标识串."""
        return decode_schema_descriptor(schema_id)


def _encode_value(value: CanonicalValue) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    items = tuple(_encode_value(item) for item in value)
    if len(items) == 1:
        return f"({items[0]},)"
    return f"({','.join(items)})"


def encode_schema_descriptor(descriptor: SchemaDescriptor) -> str:
    """校验并编码一个已登记的描述符.

    语法为 ``type_id@version(name=value,...)``. 元组、布尔、整数与字符串值各有唯一的
    规范表示; Python 模块路径、对象身份或登记顺序都不进入 ID.

    Parameters
    ----------
    descriptor : SchemaDescriptor
        在进程级 Schema 类型登记表中登记过的类型/版本对的描述符.

    Returns
    -------
    str
        确定且可逆的 Schema 标识串.

    Raises
    ------
    TypeError
        ``descriptor`` 不是 ``SchemaDescriptor``.
    ValueError
        参数约定不合法.
    KeyError
        类型 ID 或描述符版本未知.
    """
    from .registry import SCHEMA_TYPE_REGISTRY

    return SCHEMA_TYPE_REGISTRY.encode(descriptor)


def _encode_schema_descriptor_unchecked(
    descriptor: SchemaDescriptor,
) -> str:
    if not isinstance(descriptor, SchemaDescriptor):
        raise TypeError(
            "descriptor must be a SchemaDescriptor, "
            f"got {type(descriptor).__name__}"
        )

    parameters = ",".join(
        f"{name}={_encode_value(value)}"
        for name, value in descriptor.parameters
    )
    return f"{descriptor.type_id}@{descriptor.schema_version}({parameters})"


class _DescriptorParser:
    def __init__(self, schema_id: str) -> None:
        self.schema_id = schema_id
        self.position = 0

    def parse(self) -> SchemaDescriptor:
        """解析整个标识串 ``type_id@version(name=value,...)``, 返回描述符."""
        type_id = self._parse_identifier("type_id")
        self._consume("@")
        schema_version = self._parse_integer("schema_version")
        self._consume("(")

        parameters: list[tuple[str, CanonicalValue]] = []
        if not self._at(")"):
            while True:
                name = self._parse_identifier("parameter name")
                self._consume("=")
                parameters.append((name, self._parse_value()))
                if self._at(")"):
                    break
                self._consume(",")

        self._consume(")")
        if self.position != len(self.schema_id):
            self._fail("unexpected trailing content")

        return SchemaDescriptor(
            type_id=type_id,
            schema_version=schema_version,
            parameters=tuple(parameters),
        )

    def _parse_identifier(self, name: str) -> str:
        match = _IDENTIFIER_PATTERN.match(self.schema_id, self.position)
        if match is None:
            self._fail(f"expected {name}")
        self.position = match.end()
        return match.group()

    def _parse_integer(self, name: str) -> int:
        match = _INTEGER_PATTERN.match(self.schema_id, self.position)
        if match is None:
            self._fail(f"expected canonical integer for {name}")
        self.position = match.end()
        return int(match.group())

    def _parse_value(self) -> CanonicalValue:
        if self._at('"'):
            return self._parse_string()
        if self.schema_id.startswith("true", self.position):
            self.position += 4
            return True
        if self.schema_id.startswith("false", self.position):
            self.position += 5
            return False
        if self._at("("):
            return self._parse_tuple()
        return self._parse_integer("parameter value")

    def _parse_string(self) -> str:
        try:
            value, end = json.JSONDecoder().raw_decode(
                self.schema_id,
                self.position,
            )
        except json.JSONDecodeError as error:
            self._fail(f"invalid string value: {error.msg}")
        if type(value) is not str:
            self._fail("expected a string value")
        self.position = end
        return value

    def _parse_tuple(self) -> tuple[CanonicalValue, ...]:
        self._consume("(")
        if self._at(")"):
            self.position += 1
            return ()

        values = [self._parse_value()]
        if self._at(")"):
            self._fail("a one-item tuple requires a trailing comma")

        while True:
            self._consume(",")
            if self._at(")"):
                self.position += 1
                return tuple(values)
            values.append(self._parse_value())
            if self._at(")"):
                self.position += 1
                return tuple(values)

    def _consume(self, expected: str) -> None:
        if not self.schema_id.startswith(expected, self.position):
            self._fail(f"expected {expected!r}")
        self.position += len(expected)

    def _at(self, text: str) -> bool:
        return self.schema_id.startswith(text, self.position)

    def _fail(self, message: str) -> None:
        raise ValueError(
            f"invalid schema identifier at position {self.position}: {message}"
        )


def _decode_schema_descriptor_unchecked(schema_id: str) -> SchemaDescriptor:
    if not isinstance(schema_id, str):
        raise TypeError(
            f"schema_id must be a string, got {type(schema_id).__name__}"
        )

    descriptor = _DescriptorParser(schema_id).parse()
    canonical_id = _encode_schema_descriptor_unchecked(descriptor)
    if canonical_id != schema_id:
        raise ValueError(
            "schema identifier is not canonical; "
            f"expected {canonical_id!r}, got {schema_id!r}"
        )
    return descriptor


def decode_schema_descriptor(schema_id: str) -> SchemaDescriptor:
    """解码并校验一个已登记的规范 Schema 标识串.

    自定义或插件登记表可用 ``SchemaTypeRegistry.decode`` 配合各自的登记; 本便捷函数使用
    进程级的 Schema 类型登记表.

    Parameters
    ----------
    schema_id : str
        符合规范描述符语法的标识串.

    Returns
    -------
    SchemaDescriptor
        解码出的不可变描述符.

    Raises
    ------
    TypeError
        ``schema_id`` 不是字符串.
    ValueError
        标识串格式错误、不规范或参数约定不合法.
    KeyError
        类型 ID 或描述符版本未知.
    """
    from .registry import SCHEMA_TYPE_REGISTRY

    return SCHEMA_TYPE_REGISTRY.decode(schema_id)
