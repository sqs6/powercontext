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

"""Pi session history: one captured Source per visible event of the selected branch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

from pydantic import BaseModel, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME: Final[str] = "pi-event"
PI_EVENT_SOURCE_TYPE: Final[str] = "pi-event"
PI_EVENT_SOURCE_VERSION: Final[str] = "1"
PI_SESSION_CONNECTOR_NAME: Final[str] = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION: Final[str] = "1"
PI_SESSION_FORMAT_VERSION: Final[int] = 3
_PI_IDENTITY_PREFIX: Final[str] = "pi:"
_PI_CHECKPOINT_VERSION: Final[int] = 1


class PiSessionInput(BaseModel):
    """One offline Pi session archive selected for branch capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event of the selected Pi session branch."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue] = Field(default_factory=dict)


class PiEventSource(Source):
    """A captured Pi session event addressed by session identity and event identity."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue] = Field(default_factory=dict)

    materialization: SourceMaterialization = SourceMaterialization.CAPTURED


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable identity of one Pi event.

    The identity depends on the session ID and the event ID only, so the same
    event keeps one name across branches, restarts, and later appends. A long
    pair falls back to one digest so the identity never exceeds the limit.
    """

    _require_reference_part("session_id", session_id)
    _require_reference_part("event_id", event_id)
    identity = f"{_PI_IDENTITY_PREFIX}{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(f"{session_id}\x00{event_id}".encode()).hexdigest()
        identity = f"{_PI_IDENTITY_PREFIX}{digest}"
    return identity


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse one Pi session archive into the visible events of the selected branch.

    The archive is UTF-8 JSONL (a leading BOM is accepted). The first record
    declares the session, every later record joins the branch through its
    parent. Any malformed archive raises ``ValueError`` and never returns a
    partial branch.
    """

    _, events = _parse_session(value)
    return events


def _parse_session(value: PiSessionInput) -> tuple[str, tuple[PiEventInput, ...]]:
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("pi session content must be valid UTF-8") from error

    records: list[JsonValue] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"pi session line {number} is not valid JSON") from error

    if not records:
        raise ValueError("pi session must declare a session header")
    header = records[0]
    if not isinstance(header, dict):
        raise ValueError("pi session header must be a JSON object")
    if header.get("type") != "session" or header.get("version") != PI_SESSION_FORMAT_VERSION:
        raise ValueError("pi session header must declare type session and version 3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("pi session header must declare a non-empty id")

    nodes: dict[str, dict[str, JsonValue]] = {}
    order: list[str] = []
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("pi session records must be JSON objects")
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("pi session records must declare a non-empty id")
        if event_id in nodes:
            raise ValueError(f"duplicate pi session event id {event_id!r}")
        if "parentId" not in record:
            raise ValueError(f"pi session event {event_id!r} must declare parentId")
        parent_id = record.get("parentId")
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip()):
            raise ValueError(f"pi session event {event_id!r} has an invalid parentId")
        if isinstance(parent_id, str) and parent_id not in nodes:
            raise ValueError(f"pi session event {event_id!r} references an unknown parent {parent_id!r}")
        nodes[event_id] = record
        order.append(event_id)

    leaf_id = value.leaf_id
    if leaf_id is None:
        if not order:
            raise ValueError("pi session must contain at least one event")
        leaf_id = order[-1]
    elif leaf_id not in nodes:
        raise ValueError(f"pi session leaf {leaf_id!r} is missing")

    chain: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        chain.append(current)
        parent = nodes[current].get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        event = _visible_event(session_id, nodes[event_id])
        if event is not None:
            events.append(event)
    return session_id, tuple(events)


def _visible_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    """Return the visible evidence of one event, or None when it carries no text."""

    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        return None
    event_id = record.get("id")
    parent_id = record.get("parentId")

    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        body = _message_body(message)
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) else ""
    elif event_type == "custom_message":
        role = "custom"
        body = _block_body(record.get("content"))
    else:
        # Control nodes keep the graph shape but carry no visible evidence.
        return None

    if not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=str(event_id),
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _message_body(message: dict[str, JsonValue]) -> str:
    """Return the visible body of one message record."""

    role = message.get("role")
    content = message.get("content")
    if role == "toolResult":
        name = message.get("toolName")
        call_id = message.get("toolCallId")
        return (
            f"[tool_result {name if isinstance(name, str) else 'unknown'} "
            f"{call_id if isinstance(call_id, str) else 'unknown'}] {_block_body(content)}"
        )
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        body = f"$ {command}" if isinstance(command, str) else ""
        if isinstance(output, str) and output:
            body = f"{body}\n{output}" if body else output
        return body
    if isinstance(content, str):
        return content
    return _block_body(content)


def _block_body(content: JsonValue) -> str:
    """Join the visible blocks of one content value."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            payload = json.dumps(
                {} if arguments is None else arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            parts.append(
                f"[tool_call {name if isinstance(name, str) else 'unknown'} "
                f"{call_id if isinstance(call_id, str) else 'unknown'}] {payload}"
            )
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class _PiEventAdapter:
    """Resolve Pi event inputs into captured event Sources."""

    input_class: type[PiEventInput] = PiEventInput
    name: str = PI_EVENT_SOURCE_NAME
    source_class: type[PiEventSource] = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=dict(value.record),
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=dict(source.record),
        )


@dataclass(frozen=True, slots=True)
class _PiEventTextEvidenceProjection:
    """Project one captured Pi event into the standard text-evidence shape."""

    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PiEventSource] = PiEventSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: PiEventSource, /) -> object:
        evidence = TextEvidence(
            source_type=PI_EVENT_SOURCE_TYPE,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )
        return evidence.model_dump(mode="json")


PI_EVENT_SOURCE_DEFINITION: AdapterSourceDefinition[PiEventInput, PiEventSource, PiEventInput] = (
    AdapterSourceDefinition(
        adapter=_PiEventAdapter(),
        version=PI_EVENT_SOURCE_VERSION,
        projections=(_PiEventTextEvidenceProjection(),),
    )
)


class PiSessionConnector:
    """Import the visible events of one Pi session branch, once each."""

    name: Final[str] = PI_SESSION_CONNECTOR_NAME
    version: Final[str] = PI_SESSION_CONNECTOR_VERSION
    source_definitions: Final[frozenset[str]] = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        """Parse the whole archive, then submit only the events this branch adds."""

        content = self._path.read_bytes()
        session_id, events = _parse_session(PiSessionInput(content=content, leaf_id=self._leaf_id))
        imported = _imported_events(session.checkpoint)
        digests = dict(imported)
        for event in events:
            digest = _event_digest(event.record)
            known = digests.get(event.event_id)
            if known is not None:
                if known != digest:
                    raise ValueError(f"imported pi event {event.event_id!r} was rewritten")  # noqa: TRY003
                continue
            outcome = await session.submit(
                event.event_id,
                PI_EVENT_SOURCE_NAME,
                event,
            )
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE, checkpoint=session.checkpoint)
            digests[event.event_id] = digest
        checkpoint = {
            "version": _PI_CHECKPOINT_VERSION,
            "session_id": session_id,
            "events": digests,
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _imported_events(checkpoint: JsonValue | None) -> dict[str, str]:
    """Return the imported event identities and their record checksums."""

    if not isinstance(checkpoint, dict):
        return {}
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        return {}
    return {str(key): str(value) for key, value in events.items()}


def _event_digest(record: dict[str, JsonValue]) -> str:
    """Return the checksum of one original event record."""

    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_reference_part(field: str, value: str) -> None:
    """Require a non-empty reference part without leading or trailing whitespace."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")  # noqa: TRY003
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")  # noqa: TRY003


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_TYPE",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
