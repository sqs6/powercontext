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
import math
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
_SEARCH_CANDIDATE_LIMIT: Final[int] = 600
_SEARCH_QUERY_LIMIT: Final[int] = 60
_SEARCH_MATCHED_WEIGHT: Final[float] = 1.0
_SEARCH_MULTI_MATCH_BONUS: Final[float] = 2.0
_SEARCH_NUMBER_WEIGHT: Final[float] = 6.0
_SEARCH_PAGE_LIMIT: Final[int] = 4000
_SEARCH_FUSION_K: Final[int] = 10
_SEARCH_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d")
_YEAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^fy(\d{4})$")


def _reduce_word(word: str) -> str:
    """Reduce a word to the form that lets two spellings meet.

    A fiscal year is written "FY2022" in a question and "2022" in a report, and
    a question says "geographies" where the report says "geographic".
    """

    year = _YEAR_PATTERN.match(word)
    if year is not None:
        return year.group(1)
    if len(word) > 5 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


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
    citation resolves to a stored entry and its exact page Sources. A
    financial question is answered either by a table of figures or by a
    paragraph of prose, and the two are found by different signs: a table
    carries figures, a paragraph repeats the words of the question. Both
    orders are therefore produced and fused by their ranks, so a page that
    only one of them finds still reaches the result.
    """

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.artifacts.search import analyze_fts_query, analyze_text

    if limit <= 0 or not question.strip():
        return ()
    terms = tuple(dict.fromkeys(analyze_fts_query(question).split()))
    if not terms:
        return ()
    query_terms = frozenset(_reduce_word(term) for term in terms)

    listed = await memory.list()
    records = tuple(listed.entries)[:_SEARCH_PAGE_LIMIT]
    if not records:
        return ()

    page_terms: list[frozenset[str]] = []
    page_words: list[tuple[str, ...]] = []
    for record in records:
        words = tuple(analyze_text(record.entry.text).split())
        page_words.append(words)
        page_terms.append(frozenset(_reduce_word(word) for word in words))

    frequency: dict[str, int] = {}
    for tokens in page_terms:
        for token in tokens & query_terms:
            frequency[token] = frequency.get(token, 0) + 1
    size = len(page_terms)

    lexical: list[float] = []
    structural: list[float] = []
    for index, tokens in enumerate(page_terms):
        matched = query_terms & tokens
        # The lexical lane: the words of the question, rare ones first.
        score = sum(
            math.log(1 + size / (1 + frequency.get(token, 0))) for token in matched
        )
        score += _SEARCH_MATCHED_WEIGHT * len(matched)
        if len(matched) >= 2:
            score += _SEARCH_MULTI_MATCH_BONUS
        lexical.append(score)
        # The structural lane: a page that answers a financial question
        # carries figures, and the more of its words are figures the more it
        # reads like the statement that holds the answer.
        words = page_words[index]
        structural.append(
            _SEARCH_NUMBER_WEIGHT
            * (len(_SEARCH_NUMBER_PATTERN.findall(records[index].entry.text)) / max(1, len(words)))
        )

    lexical_order = sorted(range(len(records)), key=lambda index: -lexical[index])
    structural_order = sorted(range(len(records)), key=lambda index: -structural[index])
    fused: dict[int, float] = {}
    for order in (lexical_order, structural_order):
        for rank, index in enumerate(order, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (_SEARCH_FUSION_K + rank)

    citations: list[MemoryCitation] = []
    for index in sorted(fused, key=lambda item: -fused[item]):
        record = records[index]
        citations.append(record.citation)
        if len(citations) >= limit:
            break
    return tuple(citations)


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

