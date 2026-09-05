from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import Capability, CntxEvent, ContextPackage, EventType, Priority


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[A-Za-z0-9_./-]+", text.lower()) if len(term) > 2}


def _relevant_excerpt(text: str, query: str, max_chars: int) -> str:
    """Select high-overlap passages without spending a model call."""
    if len(text) <= max_chars:
        return text
    query_terms = _terms(query)
    passages = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if part.strip()]
    ranked = []
    for position, passage in enumerate(passages):
        overlap = len(query_terms & _terms(passage))
        ranked.append((overlap, -position, passage))
    ranked.sort(reverse=True)
    selected: List[tuple[int, str]] = []
    used = 0
    for overlap, negative_position, passage in ranked:
        if overlap == 0 and selected:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        selected.append((-negative_position, passage[:remaining]))
        used += min(len(passage), remaining) + 2
    if not selected:
        return text[:max_chars]
    selected.sort(key=lambda item: item[0])
    return "\n\n".join(passage for _, passage in selected)[:max_chars]


class CntxStore:
    """Append-only, hash-linked project documentation and execution memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    @staticmethod
    def _canonical(event: CntxEvent) -> bytes:
        data = event.model_dump(mode="json", exclude={"event_hash"})
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _last_hash(self) -> Optional[str]:
        last = None
        for event in self.iter_events(ignore_corrupt=False):
            last = event.event_hash
        return last

    def append(self, event: CntxEvent) -> CntxEvent:
        event.previous_hash = self._last_hash()
        event.event_hash = hashlib.sha256(self._canonical(event)).hexdigest()
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.model_dump_json(exclude_none=True) + "\n")
            handle.flush()
        return event

    def document(
        self,
        title: str,
        content: str,
        *,
        scope: str = "project",
        priority: Priority = Priority.P1,
        source: str = "user:project_documentation",
    ) -> CntxEvent:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.append(CntxEvent(
            type=EventType.DOCUMENT,
            scope=scope,
            priority=priority,
            source=source,
            summary=title,
            payload={"title": title, "content": content, "sha256": content_hash},
        ))

    def iter_events(self, *, ignore_corrupt: bool = True) -> Iterable[CntxEvent]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield CntxEvent.model_validate_json(line)
                except Exception:
                    if not ignore_corrupt:
                        raise ValueError(f"Invalid .cntx event at line {number}")

    def verify(self) -> List[str]:
        errors: List[str] = []
        previous = None
        for number, event in enumerate(self.iter_events(ignore_corrupt=False), start=1):
            if event.previous_hash != previous:
                errors.append(f"line {number}: previous hash mismatch")
            expected = hashlib.sha256(self._canonical(event)).hexdigest()
            if event.event_hash != expected:
                errors.append(f"line {number}: event hash mismatch")
            previous = event.event_hash
        return errors

    def search(
        self,
        query: str,
        *,
        scope: Optional[str] = None,
        event_types: Optional[set[EventType]] = None,
        limit: int = 12,
    ) -> List[CntxEvent]:
        query_terms = _terms(query)
        priority_weight = {Priority.P0: 5.0, Priority.P1: 3.0, Priority.P2: 2.0, Priority.P3: 1.0, Priority.P4: 0.25}
        scored = []
        p1_fallback = []
        for position, event in enumerate(self.iter_events()):
            if scope and event.scope not in {scope, "project"}:
                continue
            if event_types and event.type not in event_types:
                continue
            haystack = f"{event.summary} {json.dumps(event.payload, ensure_ascii=False)}"
            overlap = len(query_terms & _terms(haystack))
            critical = event.priority == Priority.P0
            if overlap or critical:
                scored.append((overlap * 4.0 + priority_weight[event.priority] + position / 1_000_000, event))
            elif event.priority == Priority.P1:
                p1_fallback.append((0.5 + position / 1_000_000, event))
        scored.extend(p1_fallback[-2:])
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [event for _, event in scored[:limit]]


class ContextCompiler:
    """Compiles the smallest role-specific view of `.cntx`; raw history remains intact."""

    ROLE_EVENT_TYPES = {
        Capability.MANAGER: {EventType.DOCUMENT, EventType.USER_INSTRUCTION, EventType.DECISION, EventType.OBSERVATION, EventType.TEST_RESULT, EventType.FEEDBACK, EventType.LESSON},
        Capability.SHELL: {EventType.DOCUMENT, EventType.DECISION, EventType.OBSERVATION, EventType.ERROR, EventType.FEEDBACK, EventType.LESSON},
        Capability.FILEOPS: {EventType.DOCUMENT, EventType.DECISION, EventType.OBSERVATION, EventType.PATCH, EventType.FEEDBACK, EventType.LESSON},
        Capability.CODE: {EventType.DOCUMENT, EventType.DECISION, EventType.OBSERVATION, EventType.TEST_RESULT, EventType.PATCH, EventType.FEEDBACK, EventType.LESSON},
        Capability.TEST: {EventType.DOCUMENT, EventType.OBSERVATION, EventType.TEST_RESULT, EventType.PATCH, EventType.ERROR, EventType.LESSON},
        Capability.VERIFY: {EventType.USER_INSTRUCTION, EventType.DECISION, EventType.WORKER_RESULT, EventType.OBSERVATION, EventType.TEST_RESULT, EventType.PATCH, EventType.ERROR},
        Capability.WEB_RESEARCH: {EventType.DOCUMENT, EventType.DECISION, EventType.WEB_EVIDENCE, EventType.LESSON},
    }

    def __init__(
        self,
        store: CntxStore,
        max_events: int = 12,
        episodic_memory: Any = None,
        semantic_compaction: bool = True,
    ):
        self.store = store
        self.max_events = max_events
        self.episodic_memory = episodic_memory
        self.semantic_compaction = semantic_compaction

    def _payload_for_prompt(self, event: CntxEvent, goal: str, remaining_tokens: int) -> tuple[Dict[str, Any], bool]:
        payload = dict(event.payload)
        if not self.semantic_compaction or "content" not in payload:
            return payload, False
        maximum = max(80, min(len(str(payload["content"])), max(80, remaining_tokens * 4 - 400)))
        excerpt = _relevant_excerpt(str(payload["content"]), goal, maximum)
        compacted = len(excerpt) < len(str(payload["content"]))
        payload["content"] = excerpt
        if compacted:
            payload["original_content_chars"] = len(str(event.payload["content"]))
        return payload, compacted

    def compile(
        self,
        *,
        task_id: str,
        role: Capability,
        goal: str,
        constraints: Optional[List[str]] = None,
        scope: str = "project",
        budget_tokens: int = 1800,
    ) -> ContextPackage:
        events = self.store.search(
            goal,
            scope=scope,
            event_types=self.ROLE_EVENT_TYPES.get(role),
            limit=self.max_events,
        )
        lessons: List[Dict[str, Any]] = []
        if self.episodic_memory is not None:
            lessons = self.episodic_memory.retrieve_relevant_lessons(goal, task_type=role.value, top_k=3)

        current_state: List[Dict[str, Any]] = []
        knowledge: List[Dict[str, Any]] = []
        provenance: List[str] = []
        approximate_tokens = 0
        for event in events:
            remaining_tokens = max(0, budget_tokens - approximate_tokens)
            payload, compacted = self._payload_for_prompt(event, goal, remaining_tokens)
            item = {
                "event_id": event.id,
                "type": event.type.value,
                "priority": event.priority.value,
                "summary": event.summary,
                "payload": payload,
                "compacted": compacted,
            }
            item_tokens = max(1, len(json.dumps(item, ensure_ascii=False)) // 4)
            if approximate_tokens + item_tokens > budget_tokens and event.priority != Priority.P0:
                continue
            approximate_tokens += item_tokens
            provenance.append(f"cntx:{event.id}")
            if event.type in {EventType.DOCUMENT, EventType.DECISION, EventType.LESSON, EventType.WEB_EVIDENCE}:
                knowledge.append(item)
            else:
                current_state.append(item)

        return ContextPackage(
            task_id=task_id,
            role=role,
            goal=goal,
            constraints=constraints or [],
            current_state=current_state,
            knowledge=knowledge,
            lessons=lessons,
            provenance=provenance,
            approximate_tokens=approximate_tokens,
        )
