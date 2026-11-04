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

"""Rerank coarse Memory candidates with an external HTTP scoring service."""

from __future__ import annotations

import math
from typing import Any, Final

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

HTTP_RERANK_POLICY_ID: Final[str] = "http-cross-encoder-v1"
_DEFAULT_BATCH_SIZE: Final[int] = 32
_SCORE_FIELDS: Final[frozenset[str]] = frozenset({"rank", "score"})


class HTTPMemoryReranker:
    """Select final Memory hits with an HTTP scoring service.

    The service scores a bounded batch of candidates per request, so one coarse
    pool is split into ``batch_size`` sized requests.  Every candidate keeps its
    original one-based rank inside the full pool, and the final selection is
    decided once over the scores of the whole pool rather than per batch.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        """Bind one scoring endpoint and the request batch size.

        Args:
            endpoint: URL the scoring service is reached at.
            client: client used for every scoring request, owned by the caller.
            batch_size: maximum number of candidates sent in one request.
            policy_id: identifier reported with the rerank trace.

        Raises:
            ValueError: if ``batch_size`` is not positive.
        """

        if batch_size < 1:
            raise ValueError("http reranker batch size must be positive")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the globally best original ranks scored by the service.

        Args:
            query: the query the candidates are scored against.
            candidates: the ordered coarse pool; rank ``n`` is ``candidates[n - 1]``.
            limit: how many ranks to keep.

        Returns:
            The selected original ranks and the number of requests issued.

        Raises:
            ValueError: if the bounds are invalid or a response does not match
                the documented schema.
            httpx.HTTPError: if the service answers with a non-2xx status or
                the transport fails.
        """

        _validate_bounds(len(candidates), limit)

        scores: dict[int, float] = {}
        requests = 0
        ranked = tuple(enumerate(candidates, start=1))
        for offset in range(0, len(ranked), self._batch_size):
            batch = ranked[offset : offset + self._batch_size]
            expected = {rank for rank, _ in batch}
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": rank, "text": candidate.text} for rank, candidate in batch
                    ],
                },
            )
            response.raise_for_status()
            requests += 1
            scores.update(_parse_scores(response, expected))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = tuple(rank for rank, _ in ordered[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=len(candidates) - len(selected),
        )


def _parse_scores(response: httpx.Response, expected: set[int]) -> dict[int, float]:
    """Return the scores of one batch and reject every schema deviation.

    Args:
        response: the service response of one batch.
        expected: the global ranks this batch asked for.

    Returns:
        The score of every requested rank.

    Raises:
        ValueError: if the payload, the score entries or the rank coverage do
            not match the documented schema.
    """

    payload = _response_payload(response)
    entries = payload.get("scores")
    if not isinstance(entries, list):
        raise ValueError("http reranker response must contain a scores list")  # noqa: TRY003
    parsed: dict[int, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("http reranker score entries must be objects")  # noqa: TRY003
        if set(entry) != _SCORE_FIELDS:
            raise ValueError(  # noqa: TRY003
                "http reranker score entries must contain exactly rank and score"
            )
        rank = entry["rank"]
        if type(rank) is not int:
            raise ValueError("http reranker score rank must be an integer")  # noqa: TRY003
        if rank not in expected:
            raise ValueError(  # noqa: TRY003
                "http reranker returned a rank outside the requested batch"
            )
        if rank in parsed:
            raise ValueError("http reranker returned a duplicate rank")  # noqa: TRY003
        score = entry["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("http reranker score must be a number")  # noqa: TRY003
        numeric = float(score)
        if not math.isfinite(numeric):
            raise ValueError("http reranker score must be finite")  # noqa: TRY003
        parsed[rank] = numeric
    if set(parsed) != expected:
        raise ValueError(  # noqa: TRY003
            "http reranker response must cover every requested rank exactly once"
        )
    return parsed


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    """Return the decoded response object and reject every schema deviation.

    Args:
        response: the service response of one batch.

    Returns:
        The decoded response object.

    Raises:
        ValueError: if the body is not a JSON object with exactly one
            ``scores`` field.
    """

    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("http reranker response must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("http reranker response must be a JSON object")  # noqa: TRY003
    if set(payload) != {"scores"}:
        raise ValueError(  # noqa: TRY003
            "http reranker response must contain exactly a scores field"
        )
    return payload


def _validate_bounds(candidate_count: int, limit: int) -> None:
    """Reject an empty pool and a limit outside the pool.

    Args:
        candidate_count: how many candidates the coarse pool holds.
        limit: how many ranks the caller asked to keep.

    Raises:
        ValueError: if the bounds are invalid.
    """

    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if limit < 1 or limit > candidate_count:
        raise ValueError(  # noqa: TRY003
            "rerank limit must be between one and the candidate count"
        )


__all__ = [
    "HTTP_RERANK_POLICY_ID",
    "HTTPMemoryReranker",
]
