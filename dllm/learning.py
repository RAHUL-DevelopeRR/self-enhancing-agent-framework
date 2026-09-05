from __future__ import annotations

import hashlib
import json
from typing import Any, List

from .cntx import CntxStore
from .schemas import CntxEvent, EventType, PlanStep, Priority, ToolObservation, WorkerResult


class SelfEnhancementBridge:
    """Feeds DLLM outcomes into `.cntx` and the framework's existing episode memory.

    V0 learns context and routing policies online. Weight updates are deliberately
    offline and gated; failures are preserved as future fine-tuning candidates.
    """

    def __init__(self, store: CntxStore, episodic_memory: Any = None):
        self.store = store
        self.episodic_memory = episodic_memory

    def record_step(
        self,
        *,
        task_id: str,
        step: PlanStep,
        worker_id: str,
        model: str,
        result: WorkerResult,
        observations: List[ToolObservation],
        feedback: str = "",
    ) -> float:
        tool_success = all(observation.ok and observation.data.get("exit_code", 0) == 0 for observation in observations)
        score = max(0.0, min(1.0, result.confidence * (1.0 if tool_success else 0.45)))
        success = result.status.value == "completed" and tool_success
        self.store.append(CntxEvent(
            type=EventType.WORKER_RESULT,
            scope=step.capability.value,
            priority=Priority.P2,
            source=f"model:{worker_id}",
            task_id=task_id,
            summary=result.summary,
            payload={
                "step": step.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "observations": [item.model_dump(mode="json") for item in observations],
                "score": score,
                "success": success,
            },
            provenance=result.evidence_refs,
        ))
        if feedback:
            self.store.append(CntxEvent(
                type=EventType.FEEDBACK,
                scope=step.capability.value,
                priority=Priority.P2,
                source="dllm:self_enhancement",
                task_id=task_id,
                summary=feedback,
                payload={"worker_id": worker_id, "step_id": step.id, "training_candidate": True},
            ))
        if self.episodic_memory is not None:
            fingerprint = hashlib.sha256(
                f"{result.model_dump_json()}:{feedback}".encode("utf-8")
            ).hexdigest()[:10]
            episode_id = f"{task_id}:{step.id}:{fingerprint}"
            self.episodic_memory.record_episode(
                episode_id=episode_id,
                task_type=step.capability.value,
                prompt=step.objective,
                draft=result.model_dump_json(),
                critic_feedback=feedback,
                final_content=json.dumps([item.model_dump(mode="json") for item in observations]),
                model=model,
                provider="llama.cpp",
                score=score,
                success=success,
            )
            if feedback:
                lesson_id = hashlib.sha256(f"{step.capability.value}:{feedback}".encode()).hexdigest()[:12]
                self.episodic_memory.add_or_update_lesson(
                    lesson_id=lesson_id,
                    task_type=step.capability.value,
                    trigger_pattern=step.objective[:240],
                    lesson_text=feedback[:500],
                    success=False,
                )
        return score
