# 移植自 brighthe/fealpy ``fealpy/mesh/schema/descriptor.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.

"""Stable descriptors and codecs for parameterized entity schemas."""

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
    """Describe the stable identity of one parameterized entity schema.

    Parameters are ordered pairs because their order is part of the schema
    type's versioned serialization contract. Values are restricted to
    backend-independent immutable scalars and nested tuples.

    Parameters:
        type_id: Stable registry and serialization key.
        schema_version: Positive version of the descriptor interpretation.
        parameters: Identity-bearing parameter names and normalized values in
            the order declared by the registered schema type.

    Raises:
        TypeError: If a field has an unsupported Python type.
        ValueError: If an identifier, version, or parameter layout is invalid.
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
        """Encode this descriptor as its deterministic schema identifier."""
        return encode_schema_descriptor(self)

    @classmethod
    def from_id(cls, schema_id: str) -> "SchemaDescriptor":
        """Decode an identifier using the process-wide type registry."""
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
    """Validate and encode a registered descriptor.

    The grammar is ``type_id@version(name=value,...)``. Tuple, boolean,
    integer, and string values have one canonical representation; no Python
    module path, object identity, or registry insertion order enters the ID.

    Parameters:
        descriptor: Descriptor for a type/version pair registered in the
            process-wide schema type registry.

    Returns:
        A deterministic and reversible schema identifier.

    Raises:
        TypeError: If ``descriptor`` is not a ``SchemaDescriptor``.
        ValueError: If its parameter contract is invalid.
        KeyError: If its type ID or descriptor version is unknown.
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
    """Decode and validate one registered canonical schema identifier.

    Custom or plugin registries can use ``SchemaTypeRegistry.decode`` with
    their own registrations. This convenience function uses the process-wide
    schema type registry.

    Parameters:
        schema_id: Identifier using the canonical descriptor grammar.

    Returns:
        The decoded immutable descriptor.

    Raises:
        TypeError: If ``schema_id`` is not a string.
        ValueError: If the identifier is malformed, non-canonical, or has an
            invalid parameter contract.
        KeyError: If its type ID or descriptor version is unknown.
    """
    from .registry import SCHEMA_TYPE_REGISTRY

    return SCHEMA_TYPE_REGISTRY.decode(schema_id)
