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

"""Withdraw and restore evidence across Memory and Topic Memory.

Every Scope owns one dependency graph. A Source is withdrawn with an event,
and the withdrawal propagates to the Memory entries and Topic Memory contents
that depend on it. Content whose support is gone leaves the current search
and context, keeps its history, and returns when its support returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from sqlalchemy import select, update

from powercontext.builtin.persistence.codec import dump_model, load_model, stored_bytes
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE

if TYPE_CHECKING:
    from powercontext.artifacts import ArtifactRef
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.application import BuiltinRuntime
    from powercontext.sources import SourceRef

_LIFECYCLE_BINDING = "source-lifecycle"
_LIFECYCLE_KIND = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """Report a rejected lifecycle registration or event."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason if not detail else f"{reason}: {detail}")
        self.reason = reason


class SourceLifecycleEvent(BaseModel):
    """One withdrawal or restore of a registered Source."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")  # noqa: TRY003
        return value


class LifecycleNode(BaseModel):
    """One registered node of the evidence graph."""

    family: Literal["source", "memory", "topic"]
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    reference: dict[str, JsonValue] = Field(default_factory=dict)
    revision: int = 1
    available: bool = True


class LifecycleEventRecord(BaseModel):
    """The last accepted event of one Source."""

    sequence: int
    available: bool
    reason: str


class LifecycleState(BaseModel):
    """The persisted evidence graph of one Scope."""

    nodes: dict[str, LifecycleNode] = Field(default_factory=dict)
    events: dict[str, LifecycleEventRecord] = Field(default_factory=dict)


class SourceLifecycleService:
    """Manage the evidence graph and the native state of one Scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self._scope_id = scope_id

    # ----------------------------------------------------------------- reads

    async def _load(self) -> tuple[LifecycleState, int]:
        async with self._runtime._provider.database.transaction() as connection:
            row = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_TABLE).where(
                        SOURCE_LIFECYCLE_TABLE.c.scope_id == self._scope_id
                    )
                )
            ).mappings().one_or_none()
        if row is None:
            return LifecycleState(), 0
        state = load_model(
            LifecycleState,
            stored_bytes(row["state"], column="state"),
            kind=_LIFECYCLE_KIND,
            name=_LIFECYCLE_BINDING,
        )
        return state, int(row["generation"])

    async def _save(self, state: LifecycleState, generation: int) -> int:
        payload = dump_model(state, kind=_LIFECYCLE_KIND, name=_LIFECYCLE_BINDING)
        next_generation = generation + 1
        async with self._runtime._provider.database.transaction() as connection:
            result = await connection.execute(
                update(SOURCE_LIFECYCLE_TABLE)
                .where(
                    SOURCE_LIFECYCLE_TABLE.c.scope_id == self._scope_id,
                    SOURCE_LIFECYCLE_TABLE.c.generation == generation,
                )
                .values(state=payload, generation=next_generation)
            )
            if result.rowcount == 1:
                return next_generation
            exists = (
                await connection.execute(
                    select(SOURCE_LIFECYCLE_TABLE.c.generation).where(
                        SOURCE_LIFECYCLE_TABLE.c.scope_id == self._scope_id
                    )
                )
            ).first()
            if exists is None:
                raise SourceLifecycleError("state-lost")
            raise SourceLifecycleError("state-conflict")

    async def _insert(self, state: LifecycleState) -> int:
        payload = dump_model(state, kind=_LIFECYCLE_KIND, name=_LIFECYCLE_BINDING)
        async with self._runtime._provider.database.transaction() as connection:
            await connection.execute(
                SOURCE_LIFECYCLE_TABLE.insert().values(
                    scope_id=self._scope_id,
                    state=payload,
                    generation=1,
                )
            )
        return 1

    # ---------------------------------------------------------- registrations

    async def register_source(
        self,
        node_id: str,
        source_ref: SourceRef,
        revision: int = 1,
    ) -> None:
        """Register one captured Source as the root of a dependency chain."""

        _require_text("node_id", node_id)
        if revision < 1:
            raise SourceLifecycleError("revision")
        await self._runtime.records.for_scope(self._scope_id).get_source(
            source_ref.source_type, source_ref.source_id
        )
        reference = {
            "source_type": source_ref.source_type,
            "source_id": source_ref.source_id,
        }
        await self._register(
            node_id,
            LifecycleNode(
                family="source",
                reference=reference,
                revision=revision,
                available=True,
            ),
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"] = "all",
    ) -> None:
        """Register one Memory entry with the evidence that supports it."""

        _require_text("node_id", node_id)
        entry = await self._runtime.memory.for_scope(self._scope_id).get(
            _memory_entry_request(citation)
        )
        reference = {
            "memory_ref": citation.memory_ref.model_dump(mode="json"),
            "entry_id": citation.entry_id,
            "entry_version_id": citation.entry_version_id,
        }
        del entry
        await self._register(
            node_id,
            LifecycleNode(
                family="memory",
                parents=tuple(parents),
                policy=policy,
                reference=reference,
                available=True,
            ),
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"] = "all",
    ) -> None:
        """Register one Topic Memory content with the evidence that supports it."""

        _require_text("node_id", node_id)
        record = await self._runtime.records.for_scope(self._scope_id).get_artifact_revision(
            "topic-memory", artifact_ref.artifact_id, artifact_ref.revision
        )
        reference = {"artifact_ref": artifact_ref.model_dump(mode="json")}
        del record
        await self._register(
            node_id,
            LifecycleNode(
                family="topic",
                parents=tuple(parents),
                policy=policy,
                reference=reference,
                available=True,
            ),
        )

    async def _register(self, node_id: str, node: LifecycleNode) -> None:
        state, generation = await self._load()
        existing = state.nodes.get(node_id)
        if existing is not None:
            if (
                existing.family != node.family
                or existing.reference != node.reference
                or existing.parents != node.parents
                or existing.policy != node.policy
            ):
                raise SourceLifecycleError("node-conflict", node_id)
            return
        if node.family != "source":
            if not node.parents:
                raise SourceLifecycleError("missing-parents", node_id)
            for parent in node.parents:
                if parent not in state.nodes:
                    raise SourceLifecycleError("unknown-parent", parent)
            if node_id in node.parents:
                raise SourceLifecycleError("self-cycle", node_id)
        state.nodes[node_id] = node
        _recompute(state, node_id)
        await self._persist(state, generation)

    # ------------------------------------------------------------- lifecycle

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        """Apply one withdrawal or restore and return the available nodes."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        state, generation = await self._load()
        node = state.nodes.get(event.node_id)
        if node is None or node.family != "source":
            raise SourceLifecycleError("unknown-source", event.node_id)
        if node.revision != event.revision:
            raise SourceLifecycleError("revision-mismatch", event.node_id)
        recorded = state.events.get(event.node_id)
        if recorded is not None:
            if event.sequence < recorded.sequence:
                raise SourceLifecycleError("sequence-regression", event.node_id)
            if event.sequence == recorded.sequence:
                if recorded.available == event.available and recorded.reason == event.reason:
                    return _available_nodes(state)
                raise SourceLifecycleError("sequence-conflict", event.node_id)
        previous = {node_id: node.available for node_id, node in state.nodes.items()}
        state.nodes[event.node_id] = node.model_copy(update={"available": event.available})
        state.events[event.node_id] = LifecycleEventRecord(
            sequence=event.sequence,
            available=event.available,
            reason=event.reason,
        )
        _recompute(state, event.node_id)
        await self._apply_native(previous, state, event.reason)
        await self._persist(state, generation)
        return _available_nodes(state)

    async def resolve(
        self,
        node_id: str,
        *,
        purpose: Literal["generation", "audit"],
    ) -> Any:
        """Return the original reference of one node for the requested purpose."""

        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose", str(purpose))
        state, _ = await self._load()
        node = state.nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node", node_id)
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable", node_id)
        return _reference(node)

    # ------------------------------------------------------------- native state

    async def _apply_native(
        self,
        previous: dict[str, bool],
        state: LifecycleState,
        reason: str,
    ) -> None:
        for node_id, node in state.nodes.items():
            if previous.get(node_id) == node.available:
                continue
            if node.family == "memory":
                await self._set_memory_available(node, node.available, reason)
            elif node.family == "topic":
                await self._set_topic_available(node, node.available, reason)

    async def _set_memory_available(
        self,
        node: LifecycleNode,
        available: bool,
        reason: str,
    ) -> None:
        entry_id = str(node.reference.get("entry_id", ""))
        if not entry_id:
            return
        async with self._runtime._context(self._scope_id) as context:
            service = context.artifacts.memory
            artifact_id = context.artifacts.memory_artifact_id
            try:
                current = await service.head(artifact_id)
            except Exception:  # noqa: BLE001 - a missing Memory has nothing to deactivate
                return
            entry = next(
                (item for item in await service.entries(current) if item.entry_id == entry_id),
                None,
            )
            if entry is None:
                return
            if available:
                await service.reactivate(current, entries=(entry,), reason=reason)
            else:
                await service.forget(current, entries=(entry,), reason=reason)

    async def _set_topic_available(
        self,
        node: LifecycleNode,
        available: bool,
        reason: str,
    ) -> None:
        reference = node.reference.get("artifact_ref")
        if not isinstance(reference, dict):
            return
        artifact_id = str(reference.get("artifact_id", ""))
        if not artifact_id:
            return
        records = self._runtime.records.for_scope(self._scope_id)
        if available:
            record = await records.get_artifact("topic-memory", artifact_id)
            await records.replace_artifact(
                "topic-memory",
                artifact_id,
                record.content_digest,
                _artifact_write(record.content),
            )
            return
        async with self._runtime._provider.database.transaction() as connection:
            await self._runtime._provider.repositories.topic_memories.suspend_topic(
                connection,
                self._scope_id,
                artifact_id,
            )

    async def _persist(self, state: LifecycleState, generation: int) -> None:
        try:
            await self._save(state, generation)
        except SourceLifecycleError as error:
            if error.reason != "state-lost":
                raise
            await self._insert(state)


def _memory_entry_request(citation: MemoryCitation) -> Any:
    from powercontext.builtin.runtime.models import GetMemoryEntryRequest

    return GetMemoryEntryRequest(citation=citation)


def _artifact_write(content: Any) -> Any:
    from powercontext.builtin.records import ArtifactWrite

    return ArtifactWrite(content=content)


def _reference(node: LifecycleNode) -> Any:
    from powercontext.artifacts import ArtifactRef
    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.sources import SourceRef

    if node.family == "source":
        return SourceRef(
            source_type=str(node.reference["source_type"]),
            source_id=str(node.reference["source_id"]),
        )
    if node.family == "memory":
        return MemoryCitation.model_validate(
            {
                "memory_ref": node.reference["memory_ref"],
                "entry_id": node.reference["entry_id"],
                "entry_version_id": node.reference["entry_version_id"],
            }
        )
    return ArtifactRef.model_validate(node.reference["artifact_ref"])


def _recompute(state: LifecycleState, changed: str) -> None:
    """Recompute the derived availability of every node of the graph."""

    order = _topological_order(state, changed)
    for node_id in order:
        node = state.nodes[node_id]
        if node.family == "source":
            continue
        support = [state.nodes[parent].available for parent in node.parents if parent in state.nodes]
        if not support:
            node.available = False
        elif node.policy == "all":
            node.available = all(support)
        else:
            node.available = any(support)


def _topological_order(state: LifecycleState, changed: str) -> list[str]:
    """Return the node ids in an order where every parent precedes its children."""

    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise SourceLifecycleError("cycle", node_id)
        visiting.add(node_id)
        node = state.nodes.get(node_id)
        if node is not None:
            for parent in node.parents:
                if parent in state.nodes:
                    visit(parent)
        visiting.discard(node_id)
        visited.add(node_id)
        ordered.append(node_id)

    visit(changed)
    for node_id in state.nodes:
        visit(node_id)
    return ordered


def _available_nodes(state: LifecycleState) -> list[str]:
    return sorted(node_id for node_id, node in state.nodes.items() if node.available)


def _require_text(field: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError(field)
    if value != value.strip():
        raise SourceLifecycleError(field)


__all__ = [
    "LifecycleNode",
    "LifecycleState",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleService",
]
