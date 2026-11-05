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

"""Page-level capture of one financial report PDF.

Every physical page of the report becomes one captured Source, so a question
answer can be checked against the exact page it came from. The page number
counts physical pages from the first page of the file.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import MemoryCitation

PDF_PAGE_SOURCE_NAME: Final[str] = "pdf-page"
PDF_PAGE_SOURCE_TYPE: Final[str] = "pdf-page"
PDF_PAGE_SOURCE_VERSION: Final[str] = "1"
_PDF_IDENTITY_PREFIX: Final[str] = "pdf:"
_SEARCH_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
_SEARCH_CANDIDATE_LIMIT: Final[int] = 100


class PDFDocumentInput(BaseModel):
    """One offline PDF document selected for page-level capture."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be empty or whitespace-only")
        return value


class PDFPageInput(BaseModel):
    """One physical page of a PDF document with its document digest."""

    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str


class PDFPageSource(Source):
    """A captured PDF page addressed by document, digest, and physical page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int = Field(ge=1)
    text: str

    materialization: SourceMaterialization = SourceMaterialization.CAPTURED


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable identity of one PDF page.

    The identity depends on the document ID, the digest of the whole file, and
    the physical page number, so a revised document produces new identities
    while the older references keep pointing at the older revision.
    """

    if not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace-only")  # noqa: TRY003
    identity = f"{_PDF_IDENTITY_PREFIX}{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(
            f"{document_id}\x00{document_sha256}\x00{page_number}".encode()
        ).hexdigest()
        identity = f"{_PDF_IDENTITY_PREFIX}{digest}"
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse one PDF document into its physical pages.

    The whole file is parsed before any page is returned, so a broken document
    fails with ``ValueError`` instead of returning a partial document. A
    permission-only PDF that needs no user password is valid input.
    """

    document_id = value.document_id
    if not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace-only")  # noqa: TRY003
    content = value.content
    if not content:
        raise ValueError("pdf document must not be empty")  # noqa: TRY003
    digest = hashlib.sha256(content).hexdigest()

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            # An empty user password only carries permission flags.
            if not reader.decrypt(""):
                raise ValueError("pdf document needs a password")  # noqa: TRY003
        pages: list[PDFPageInput] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as error:  # noqa: BLE001 - any broken page fails the document
                raise ValueError("pdf page could not be extracted") from error
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=digest,
                    page_number=number,
                    text=text,
                )
            )
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - any unreadable document fails atomically
        raise ValueError("pdf document could not be parsed") from error
    return tuple(pages)


@dataclass(frozen=True, slots=True)
class _PDFPageAdapter:
    """Resolve PDF page inputs into captured page Sources."""

    input_class: type[PDFPageInput] = PDFPageInput
    name: str = PDF_PAGE_SOURCE_NAME
    source_class: type[PDFPageSource] = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


@dataclass(frozen=True, slots=True)
class _PDFPageTextEvidenceProjection:
    """Project one captured PDF page into the standard text-evidence shape."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PDFPageSource] = PDFPageSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: PDFPageSource, /) -> object:
        evidence = TextEvidence(
            source_type=PDF_PAGE_SOURCE_TYPE,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )
        return evidence.model_dump(mode="json")


PDF_PAGE_SOURCE_DEFINITION: AdapterSourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = (
    AdapterSourceDefinition(
        adapter=_PDFPageAdapter(),
        version=PDF_PAGE_SOURCE_VERSION,
        projections=(_PDFPageTextEvidenceProjection(),),
    )
)


async def search_pdf_pages(
    memory: object,
    question: str,
    limit: int = 10,
) -> tuple[MemoryCitation, ...]:
    """Return the page citations that answer one question, most relevant first.

    The pages are read through the public Memory API, so every returned
    citation resolves to a stored entry and its exact page Sources. The pages
    are then reordered by how much of the question they cover, because the
    lexical index alone ranks by term frequency.
    """

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    if limit <= 0 or not question.strip():
        return ()
    page = await memory.search(
        SearchMemoryRequest(query=question, mode="fts", limit=max(limit, _SEARCH_CANDIDATE_LIMIT))
    )
    terms = _search_terms(question)
    scored: list[tuple[int, int, object]] = []
    for rank, hit in enumerate(page.hits):
        matched = len(terms & _search_terms(hit.text))
        scored.append((-matched, rank, hit))
    scored.sort(key=lambda item: (item[0], item[1]))
    citations: list[MemoryCitation] = []
    seen: set[tuple[str, str]] = set()
    for _, _, hit in scored:
        identity = (hit.entry_id, hit.entry_version_id)
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
        if len(citations) >= limit:
            break
    return tuple(citations)


def _search_terms(text: str) -> frozenset[str]:
    return frozenset(_SEARCH_TOKEN_PATTERN.findall(text.casefold()))


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_TYPE",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
