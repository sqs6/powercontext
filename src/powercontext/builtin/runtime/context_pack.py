# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context Packs: transport one prepared context, then only what changed."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

CONTEXT_PACK_SCHEMA: Final[str] = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA: Final[str] = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES: Final[int] = 512

_PACK_FIELDS: Final[frozenset[str]] = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
)


class ContextPackError(ValueError):
    """A context pack or delta broke the wire contract."""


@dataclass(frozen=True)
class ContextPack:
    """One prepared context split into bounded, checksummed chunks.

    The wire form carries the size of the whole payload, so the byte count has
    to describe itself.  That is a fixed point: it is resolved once, when the
    pack is serialized.
    """

    items: tuple[bytes, ...]

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Return a pack over the given chunks.

        Args:
            items: the chunks of the context.

        Returns:
            The pack over those chunks.

        Raises:
            ContextPackError: if a chunk is not bytes.
        """

        chunks = tuple(items)
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise ContextPackError("context pack items must be bytes")  # noqa: TRY003
        return cls(chunks)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Return a deterministic pack over the given text.

        Every chunk holds whole characters only and stays within ``chunk_bytes``,
        so the chunks can be concatenated back into the original bytes.

        Args:
            content: the text to split.
            chunk_bytes: the maximum size of one chunk in bytes.

        Returns:
            The pack over the split text.

        Raises:
            ContextPackError: if the limit is invalid or too small for one
                character.
        """

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("context pack chunk size must be a positive integer")  # noqa: TRY003
        chunks: list[bytes] = []
        current = bytearray()
        for character in content:
            encoded = character.encode("utf-8")
            if len(current) + len(encoded) > chunk_bytes:
                if not current:
                    raise ContextPackError(  # noqa: TRY003
                        "context pack chunk size cannot hold one character"
                    )
                chunks.append(bytes(current))
                current = bytearray()
            current += encoded
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode one pack and check it against the wire contract.

        Args:
            data: the serialized pack.

        Returns:
            The decoded pack.

        Raises:
            ContextPackError: if the payload is malformed, its digest does not
                match its chunks or its byte count is wrong.
        """

        payload = _decode_object(data, _PACK_FIELDS)
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("context pack schema is not supported")  # noqa: TRY003
        pack = cls.build(_decode_items(payload["items"]))
        if payload["digest"] != pack.digest:
            raise ContextPackError("context pack digest does not match its items")  # noqa: TRY003
        if payload["byte_count"] != pack.byte_count:
            raise ContextPackError("context pack byte count does not match the payload")  # noqa: TRY003
        return pack

    @property
    def digest(self) -> str:
        """Return the digest of the schema and the chunks."""

        return "sha256:" + hashlib.sha256(_wire_bytes(self._unsigned())).hexdigest()

    @property
    def byte_count(self) -> int:
        """Return the size of the serialized pack, including the byte count field."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the pack."""

        payload = {**self._unsigned(), "digest": self.digest}
        return _wire_bytes({**payload, "byte_count": _resolve_byte_count(payload)})

    def _unsigned(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": [_encode_item(item) for item in self.items],
        }


@dataclass(frozen=True)
class ContextPackDelta:
    """The difference between one context pack and the next one.

    The receiver keeps ``prefix_items`` chunks of the base pack and appends
    ``appended`` to obtain the target pack.
    """

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Return the delta that turns the base pack into the target pack.

        The longest common chunk prefix is kept; the rest of the target is
        appended.

        Args:
            base: the pack the caller already holds.
            target: the pack that should be reconstructed.

        Returns:
            The delta between the two packs.
        """

        limit = min(len(base.items), len(target.items))
        prefix = 0
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode one delta and check it against the wire contract.

        Args:
            data: the serialized delta.

        Returns:
            The decoded delta.

        Raises:
            ContextPackError: if the payload is malformed or the prefix is not
                a non-negative integer.
        """

        payload = _decode_object(data, _DELTA_FIELDS)
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("context pack delta schema is not supported")  # noqa: TRY003
        prefix = payload["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError(  # noqa: TRY003
                "context pack delta prefix must be a non-negative integer"
            )
        for name in ("base_digest", "target_digest"):
            value = payload[name]
            if not isinstance(value, str) or not value:
                raise ContextPackError(  # noqa: TRY003
                    "context pack delta digest must be a non-empty string"
                )
        return cls(
            payload["base_digest"],
            payload["target_digest"],
            prefix,
            _decode_items(payload["appended"]),
        )

    @property
    def byte_count(self) -> int:
        """Return the size of the serialized delta."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta."""

        return _wire_bytes(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_encode_item(item) for item in self.appended],
            }
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from the base pack and one delta.

    Args:
        base: the pack the receiver already holds.
        delta: the delta describing the change.

    Returns:
        The reconstructed target pack.

    Raises:
        ContextPackError: if the delta does not belong to the base pack, uses a
            prefix outside the base pack, or does not rebuild the target.
    """

    if base.digest != delta.base_digest:
        raise ContextPackError("context pack delta does not match the base pack")  # noqa: TRY003
    if delta.prefix_items < 0 or delta.prefix_items > len(base.items):
        raise ContextPackError("context pack delta prefix is out of range")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError(  # noqa: TRY003
            "context pack delta does not reconstruct the target pack"
        )
    return target


def _wire_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize one payload in the canonical wire form."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_item(item: bytes) -> str:
    """Return the base64 form of one chunk."""

    return base64.b64encode(item).decode("ascii")


def _resolve_byte_count(payload: dict[str, Any]) -> int:
    """Return the byte count that describes a payload including its own field.

    The wire size grows by one digit when the count grows by one, so the fixed
    point is found by iterating until the value stops moving.

    Args:
        payload: the payload without its byte count field.

    Returns:
        The byte count the serialized payload has.

    Raises:
        ContextPackError: if the value does not settle.
    """

    count = 0
    for _ in range(32):
        sized = len(_wire_bytes({**payload, "byte_count": count}))
        if sized == count:
            return count
        count = sized
    raise ContextPackError("context pack byte count does not converge")  # noqa: TRY003


def _decode_object(data: bytes, expected: frozenset[str]) -> dict[str, Any]:
    """Decode one wire object and check that it holds exactly the expected fields.

    Args:
        data: the serialized payload.
        expected: the field names the schema allows.

    Returns:
        The decoded payload.

    Raises:
        ContextPackError: if the payload is not a JSON object with exactly the
            expected fields.
    """

    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContextPackError("context pack payload must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("context pack payload must be a JSON object")  # noqa: TRY003
    if set(payload) != expected:
        raise ContextPackError(  # noqa: TRY003
            "context pack payload fields do not match the schema"
        )
    return payload


def _decode_items(values: Any) -> tuple[bytes, ...]:
    """Decode the base64 chunk list of one payload.

    Args:
        values: the decoded JSON value holding the list.

    Returns:
        The decoded chunks.

    Raises:
        ContextPackError: if the value is not a list of base64 strings.
    """

    if not isinstance(values, list):
        raise ContextPackError("context pack items must be a list")  # noqa: TRY003
    chunks: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("context pack items must be base64 strings")  # noqa: TRY003
        try:
            chunks.append(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError("context pack items must be valid base64") from error
    return tuple(chunks)


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
