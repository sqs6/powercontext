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

"""Coalesce concurrent embedding requests and reuse successful vectors.

Callers that need the same text inside one merge window share one
underlying request, and every successful vector is kept in a bounded LRU
cache for later calls. Cancelling one caller never cancels the shared
underlying work, and a failed request is never cached.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass
class _PendingText:
    """One text whose vector is still being computed."""

    key: tuple[str, str]
    text: str
    future: asyncio.Future[EmbeddingVector] = field(repr=False)
    batch_id: int | None = None
    usage: InferenceUsage | None = None
    dispatched: bool = False


class CoalescingEmbeddingModel:
    """Share concurrent embedding work and reuse the vectors that succeeded."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str], EmbeddingVector] = OrderedDict()
        self._pending: dict[tuple[str, str], _PendingText] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._next_batch_id = 0
        if hasattr(model, "embed_query"):
            # A provider that separates queries from documents keeps that
            # distinction; the cache stays separate so the two never mix.
            self.embed_query = self._embed_query  # type: ignore[method-assign]

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input text, in the input order."""

        return await self._embed("embed", texts)

    async def _embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self._embed("query", texts)

    async def _embed(self, kind: str, texts: tuple[str, ...]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        vectors: dict[str, EmbeddingVector] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self._cached(kind, text)
            if cached is None:
                missing.append(text)
            else:
                vectors[text] = cached
        if not missing:
            return EmbeddingResult(
                vectors=tuple(vectors[text] for text in texts),
                usage=InferenceUsage(requests=0),
            )

        entries: list[_PendingText] = []
        for text in missing:
            key = (kind, text)
            entry = self._pending.get(key)
            if entry is None:
                entry = _PendingText(
                    key=key,
                    text=text,
                    future=asyncio.get_running_loop().create_future(),
                )
                self._pending[key] = entry
            entries.append(entry)
        self._ensure_dispatcher()

        # Waiting on the shared futures never cancels them, so one cancelled
        # caller cannot take the shared work down with it.
        await asyncio.wait([entry.future for entry in entries])
        for entry in entries:
            error = entry.future.exception()
            if error is not None:
                raise error
            vectors[entry.text] = entry.future.result()
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=self._usage(entries),
        )

    def _usage(self, entries: list[_PendingText]) -> InferenceUsage:
        batches: dict[int, InferenceUsage | None] = {}
        for entry in entries:
            if entry.batch_id is not None:
                batches.setdefault(entry.batch_id, entry.usage)
        usages = tuple(usage for usage in batches.values() if usage is not None)
        return InferenceUsage(
            requests=len(batches),
            input_tokens=_sum_usage(usages, "input_tokens"),
            output_tokens=_sum_usage(usages, "output_tokens"),
        )

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch_loop(), name="embedding-coalescer")

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                await self._wait_window()
                entries = [entry for entry in self._pending.values() if not entry.dispatched]
                if not entries:
                    if not self._pending:
                        return
                    continue
                for entry in entries:
                    entry.dispatched = True
                batches = [
                    entries[index : index + self._max_batch_size]
                    for index in range(0, len(entries), self._max_batch_size)
                ]
                await asyncio.gather(
                    *(self._run_batch(batch) for batch in batches),
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            raise

    async def _wait_window(self) -> None:
        if self._coalesce_seconds > 0:
            await asyncio.sleep(self._coalesce_seconds)
        else:
            await asyncio.sleep(0)

    async def _run_batch(self, entries: list[_PendingText]) -> None:
        kind = entries[0].key[0]
        texts = tuple(entry.text for entry in entries)
        sender = self._model.embed if kind == "embed" else self._model.embed_query
        self._next_batch_id += 1
        batch_id = self._next_batch_id
        for entry in entries:
            entry.batch_id = batch_id
        try:
            result = await sender(texts)
        except BaseException as error:  # noqa: BLE001 - the waiters see the original failure
            self._fail(entries, error)
            return
        vectors = tuple(result.vectors)
        if len(vectors) != len(entries):
            self._fail(entries, ValueError("embedding model returned an unexpected number of vectors"))
            return
        for entry, vector in zip(entries, vectors, strict=True):
            entry.usage = result.usage
            self._store(kind, entry.text, vector)
            if not entry.future.done():
                entry.future.set_result(vector)
        for entry in entries:
            self._pending.pop(entry.key, None)

    def _fail(self, entries: list[_PendingText], error: BaseException) -> None:
        for entry in entries:
            if not entry.future.done():
                entry.future.set_exception(error)
            self._pending.pop(entry.key, None)

    def _cached(self, kind: str, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        key = (kind, text)
        vector = self._cache.get(key)
        if vector is None:
            return None
        self._cache.move_to_end(key)
        return vector

    def _store(self, kind: str, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        key = (kind, text)
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _sum_usage(usages: tuple[InferenceUsage, ...], field: str) -> int | None:
    if not usages:
        return None
    values = [getattr(usage, field) for usage in usages]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


__all__ = ["CoalescingEmbeddingModel"]
