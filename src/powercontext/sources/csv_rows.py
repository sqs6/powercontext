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

"""Row-level CSV capture: one captured Source per record, addressable by primary key."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel, field_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.definitions import AdapterSourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME: Final[str] = "csv-row"
CSV_ROW_SOURCE_VERSION: Final[str] = "1"
CSV_ROW_SOURCE_TYPE: Final[str] = "csv-row"
_CSV_IDENTITY_PREFIX: Final[str] = "csv:"


class CSVDocumentInput(BaseModel):
    """One offline CSV document selected for row-level capture."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV record with its original column order and raw field values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured CSV record addressed by document identity and primary key value."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    materialization: SourceMaterialization = SourceMaterialization.CAPTURED

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        _require_reference_part("source_id", value)
        return value


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable identity of one CSV row.

    The identity is ``csv:{percent-encoded document_id}:{percent-encoded primary_key_value}``.
    Both components are percent-encoded with ``safe=""`` so that no separator can collide.
    """

    _require_reference_part("document_id", document_id)
    if not primary_key_value.strip():
        raise ValueError("primary_key_value must not be empty or whitespace-only")
    identity = (
        f"{_CSV_IDENTITY_PREFIX}{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(
            f"csv row source id must not exceed {MAX_SOURCE_ID_LENGTH} characters"
        )
    return identity


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse one offline CSV document into row-level inputs.

    The document is decoded as strict UTF-8 (a leading BOM is accepted) and parsed with
    ``csv.reader`` semantics in strict mode.  The first record is the header.  Every
    validation error is raised as ``ValueError`` and never yields partial rows.
    """

    document_id = value.document_id
    primary_key = value.primary_key

    _require_reference_part("document_id", document_id)
    if not primary_key.strip():
        raise ValueError("primary_key must not be empty or whitespace-only")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("csv content must be valid UTF-8") from error

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        records = list(reader)
    except csv.Error as error:
        raise ValueError(f"csv content is malformed: {error}") from error

    if not records:
        raise ValueError("csv document must contain a header row")

    header = tuple(records[0])
    if not header:
        raise ValueError("csv header row must not be empty")
    for column in header:
        if not column or not column.strip():
            raise ValueError("csv header columns must be non-empty and not whitespace-only")
    if len(set(header)) != len(header):
        raise ValueError("csv header columns must be unique")
    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} is not one of the csv header columns")

    primary_key_index = header.index(primary_key)
    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records[1:]:
        if not record:
            # A completely empty physical record carries no fields and is ignored.
            continue
        if len(record) != len(header):
            raise ValueError("csv record width does not match the header width")
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("csv primary key value must not be empty or whitespace-only")
        if primary_key_value in seen:
            raise ValueError(f"duplicate csv primary key value {primary_key_value!r}")
        seen.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=header,
                values=tuple(record),
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _CSVRowAdapter:
    """Resolve CSV row inputs into captured row Sources."""

    input_class: type[CSVRowInput] = CSVRowInput
    name: str = CSV_ROW_SOURCE_NAME
    source_class: type[CSVRowSource] = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=tuple(value.columns),
            values=tuple(value.values),
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=tuple(source.columns),
            values=tuple(source.values),
        )


@dataclass(frozen=True, slots=True)
class _CSVRowTextEvidenceProjection:
    """Project a captured CSV row into the standard text-evidence shape."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[CSVRowSource] = CSVRowSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: CSVRowSource, /) -> object:
        row = dict(zip(source.columns, source.values, strict=True))
        evidence = TextEvidence(
            source_type=CSV_ROW_SOURCE_TYPE,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )
        return evidence.model_dump(mode="json")


CSV_ROW_SOURCE_DEFINITION: AdapterSourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = (
    AdapterSourceDefinition(
        adapter=_CSVRowAdapter(),
        version=CSV_ROW_SOURCE_VERSION,
        projections=(_CSVRowTextEvidenceProjection(),),
    )
)


def _require_reference_part(field: str, value: str) -> None:
    """Require a non-empty reference part without leading or trailing whitespace."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_TYPE",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
