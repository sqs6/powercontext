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

"""Count admission for Source-driven Families.

The pending count of one Scope and Family is the number of journal entries
published after the Source cursor that the Family last committed. A threshold
of None keeps the count condition off, so the time condition stays alone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext._logging import log_safely
from powercontext.builtin.persistence.cursors import SourceCursorRepository
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.processing_intents import ArtifactProcessingIntentRepository
from powercontext.builtin.persistence.tables import SOURCE_JOURNAL_HEADS_TABLE

logger = logging.getLogger(__name__)

COUNT_TRIGGER_POLL_SECONDS = 0.2


class SourceCountTrigger:
    """Accept one automatic processing call when pending Source entries reach a threshold."""

    def __init__(
        self,
        database: AsyncDatabase,
        *,
        binding_name: str,
        threshold: int,
        prepare: Callable[[AsyncConnection, str], Awaitable[bool]] | None = None,
        intents: ArtifactProcessingIntentRepository | None = None,
        cursors: SourceCursorRepository | None = None,
    ) -> None:
        if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 1:
            raise ValueError("Source count threshold must be a positive integer")  # noqa: TRY003
        self._database = database
        self._binding_name = binding_name
        self._threshold = threshold
        self._prepare = prepare
        self._intents = intents or ArtifactProcessingIntentRepository()
        self._cursors = cursors or SourceCursorRepository()

    @property
    def binding_name(self) -> str:
        return self._binding_name

    @property
    def threshold(self) -> int:
        return self._threshold

    async def pending_counts(self) -> dict[str, int]:
        """Return the pending Source count of every Scope that owns a journal."""

        async with self._database.transaction() as connection:
            return await self._counts(connection)

    async def _counts(self, connection: AsyncConnection) -> dict[str, int]:
        heads = SOURCE_JOURNAL_HEADS_TABLE
        rows = (await connection.execute(select(heads.c.scope_id, heads.c.position))).all()
        counts: dict[str, int] = {}
        for scope, high in rows:
            scope_id = str(scope)
            cursor = await self._cursors.load(connection, scope_id, self._binding_name)
            counts[scope_id] = int(high) - (0 if cursor is None else cursor.cursor.sequence)
        return counts

    async def scan(self) -> tuple[str, ...]:
        """Request one processing call for every Scope whose pending count reaches the threshold.

        A Scope keeps its accepted request until the Worker finishes it, so a
        running window is never requested twice. The count is read from the
        committed cursor, never from the body length or the generated result.
        """

        async with self._database.transaction() as connection:
            counts = await self._counts(connection)
            requested: list[str] = []
            for scope_id, pending in counts.items():
                if pending < self._threshold:
                    continue
                intent = await self._intents.load(connection, scope_id, self._binding_name)
                if intent is not None and intent.requested_generation > intent.handled_generation:
                    # A request is still open, so this Scope is already being
                    # worked on and must not be requested a second time.
                    continue
                if self._prepare is not None and not await self._prepare(connection, scope_id):
                    continue
                await self._intents.request(connection, scope_id, self._binding_name)
                requested.append(scope_id)
        return tuple(requested)


class SourceCountTriggerGroup:
    """Run the count triggers of one Runtime and wake their Supervisor after each request."""

    def __init__(
        self,
        triggers: Sequence[SourceCountTrigger],
        *,
        wake: Callable[[str], None],
        interval_seconds: float = COUNT_TRIGGER_POLL_SECONDS,
    ) -> None:
        self._triggers = tuple(triggers)
        self._wake = wake
        self._interval = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def __aenter__(self) -> SourceCountTriggerGroup:
        if self._triggers:
            self._task = asyncio.create_task(self._run(), name="source-count-trigger")
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                log_safely(
                    logger,
                    logging.WARNING,
                    "Source count admission failed",
                    exc_info=error,
                    extra={"event": "source_count_trigger.failed"},
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    async def _scan_once(self) -> None:
        for trigger in self._triggers:
            scopes = await trigger.scan()
            if scopes:
                self._wake(trigger.binding_name)


__all__ = ["COUNT_TRIGGER_POLL_SECONDS", "SourceCountTrigger", "SourceCountTriggerGroup"]
