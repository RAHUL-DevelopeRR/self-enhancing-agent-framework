from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from .config import DLLMConfig, WorkerConfig
from .cntx import CntxStore, ContextCompiler
from .learning import SelfEnhancementBridge
from .runtime import LlamaCppTransport, ModelTransport
from .schemas import (
    Capability,
    CntxEvent,
    EventType,
    PlanStep,
    Priority,
    RunReport,
    StepReport,
    TaskPlan,
    ToolObservation,
    WorkOrder,
    WorkerResult,
    WorkerStatus,
)
from .tools import SafeToolExecutor
from .web_rag import SearxngWebRag


class DLLMOrchestrator:
    """Manager/worker organization with phase-gated model activation.

    The manager plans and resolves ambiguity. Specialists receive bounded work
    orders and `.cntx` packages. Deterministic tools own real side effects.
    """

    def __init__(
        self,
        config: DLLMConfig,
        project_root: str | Path,
        *,
        transport: Optional[ModelTransport] = None,
        store: Optional[CntxStore] = None,
        episodic_memory=None,
    ):
        self.config = config
        self.project_root = Path(project_root).resolve()
        context_path = Path(config.context.path)
        if not context_path.is_absolute():
            context_path = self.project_root / context_path
        self.store = store or CntxStore(context_path)
        self.compiler = ContextCompiler(
            self.store,
            max_events=config.context.max_events_per_package,
            episodic_memory=episodic_memory,
            semantic_compaction=config.context.semantic_compaction,
        )
        self.transport = transport or LlamaCppTransport()
        self.workers = config.workers_by_capability()
        self.executor = SafeToolExecutor(self.project_root, config.tools)
        self.learning = SelfEnhancementBridge(self.store, episodic_memory)
        self.web_rag = SearxngWebRag(config.web_rag, self.store)

    def _worker(self, capability: Capability) -> WorkerConfig:
        worker = self.workers.get(capability)
        if worker is None:
            raise RuntimeError(f"No enabled worker for capability '{capability.value}'")
        return worker

    @staticmethod
    def _fast_path(goal: str) -> Optional[Capability]:
        normalized = goal.strip().lower()
        if normalized.startswith(("list files", "find files", "search files", "read file", "show file")):
            return Capability.FILEOPS
        if normalized.startswith(("run tests", "run test", "test the", "execute tests")):
            return Capability.TEST
        if normalized.startswith(("bash ", "powershell ", "shell command", "run command")):
            return Capability.SHELL
        if normalized.startswith(("search web", "research web", "look up online")):
            return Capability.WEB_RESEARCH
        return None

    async def _manager_plan(self, task_id: str, goal: str, failure_context: str = "") -> TaskPlan:
        manager = self._worker(Capability.MANAGER)
        context = self.compiler.compile(
            task_id=task_id,
            role=Capability.MANAGER,
            goal=goal,
            budget_tokens=manager.context_budget,
        )
        manifest = [
            {"worker_id": worker.id, "capability": worker.capability.value}
            for worker in self.workers.values()
            if worker.capability not in {Capability.MANAGER, Capability.VERIFY}
        ]
        plan = await self.transport.generate(manager, {
            "goal": goal,
            "failure_context": failure_context,
            "worker_manifest": manifest,
            "project_context": context.model_dump(mode="json"),
            "rules": {
                "manager_does_not_execute": True,
                "ask_on_material_user_ambiguity": True,
                "prefer_parallel_read_only_observation": True,
                "max_steps": 8,
            },
            "response_schema": TaskPlan.model_json_schema(),
        }, TaskPlan)
        if len(plan.steps) > 8:
            raise RuntimeError("Manager plan exceeded the V0 eight-step safety limit")
        forbidden = [step.id for step in plan.steps if step.capability in {Capability.MANAGER, Capability.VERIFY}]
        if forbidden:
            raise RuntimeError(f"Manager delegated reserved capabilities in steps: {forbidden}")
        return plan

    async def _initial_plan(self, task_id: str, goal: str) -> Tuple[TaskPlan, bool]:
        capability = self._fast_path(goal) if self.config.routing.fast_path else None
        if capability and capability in self.workers:
            return TaskPlan(
                goal=goal,
                summary="Deterministic fast-path delegation; manager reserved for ambiguity.",
                steps=[PlanStep(
                    id="fast-1",
                    capability=capability,
                    objective=goal,
                    constraints=["Stay within the project root", "Report ambiguity instead of guessing"],
                    acceptance_criteria=["Return evidence from the real environment"],
                )],
            ), True
        return await self._manager_plan(task_id, goal), False

    async def _invoke_worker(self, task_id: str, step: PlanStep, feedback: Optional[List[str]] = None) -> Tuple[WorkerConfig, WorkOrder, WorkerResult]:
        worker = self._worker(step.capability)
        if step.capability == Capability.WEB_RESEARCH:
            await self.web_rag.retrieve(step.objective, task_id)
        context = self.compiler.compile(
            task_id=task_id,
            role=step.capability,
            goal=step.objective,
            constraints=step.constraints,
            budget_tokens=worker.context_budget,
        )
        order = WorkOrder(
            task_id=task_id,
            step_id=step.id,
            capability=step.capability,
            objective=step.objective,
            constraints=step.constraints,
            acceptance_criteria=step.acceptance_criteria,
            available_tools=self.executor.available_for(step.capability.value),
            context=context.model_dump(mode="json"),
            feedback=feedback or [],
        )
        self.store.append(CntxEvent(
            type=EventType.WORK_ORDER,
            scope=step.capability.value,
            priority=Priority.P2,
            source="dllm:manager",
            task_id=task_id,
            summary=step.objective,
            payload=order.model_dump(mode="json"),
            provenance=context.provenance,
        ))
        result = await self.transport.generate(worker, {
            "work_order": order.model_dump(mode="json"),
            "response_schema": WorkerResult.model_json_schema(),
        }, WorkerResult)
        if result.step_id != step.id:
            result.step_id = step.id
        return worker, order, result

    def _execute_actions(self, capability: Capability, result: WorkerResult) -> List[ToolObservation]:
        allowed = set(self.executor.available_for(capability.value))
        observations: List[ToolObservation] = []
        for call in result.tool_calls:
            if call.name not in allowed:
                observations.append(ToolObservation(
                    tool=call.name,
                    ok=False,
                    summary=f"Tool '{call.name}' is outside the {capability.value} worker boundary",
                ))
            else:
                observations.append(self.executor.execute(call))
        return observations

    async def _verify(self, task_id: str, order: WorkOrder, result: WorkerResult, observations: List[ToolObservation]) -> Optional[WorkerResult]:
        verifier = self.workers.get(Capability.VERIFY)
        if verifier is None:
            return None
        allowed = set(self.executor.available_for(order.capability.value))
        mutated = self.executor.has_side_effects(call for call in result.tool_calls if call.name in allowed)
        if not mutated or not self.config.routing.verify_all_mutations:
            return None
        return await self.transport.generate(verifier, {
            "work_order": order.model_dump(mode="json"),
            "candidate": result.model_dump(mode="json"),
            "real_observations": [item.model_dump(mode="json") for item in observations],
            "response_schema": WorkerResult.model_json_schema(),
        }, WorkerResult)

    def _feedback(self, result: WorkerResult, observations: List[ToolObservation], verification: Optional[WorkerResult]) -> str:
        if result.status != WorkerStatus.COMPLETED:
            detail = result.ambiguity or "; ".join(result.missing_information) or result.summary
            return f"Worker status {result.status.value}: {detail}"
        if result.confidence < self.config.routing.confidence_threshold:
            return (
                f"Worker confidence {result.confidence:.2f} is below the calibrated "
                f"threshold {self.config.routing.confidence_threshold:.2f}"
            )
        failures = [item.summary for item in observations if not item.ok or item.data.get("exit_code", 0) != 0]
        if failures:
            return "Real execution evidence failed: " + " | ".join(failures)
        if verification and verification.status != WorkerStatus.COMPLETED:
            return "Verifier rejected the result: " + verification.summary
        return ""

    async def _run_step(self, task_id: str, step: PlanStep) -> StepReport:
        started = time.perf_counter()
        feedback_items: List[str] = []
        attempts = 0
        final_worker: Optional[WorkerConfig] = None
        final_result: Optional[WorkerResult] = None
        final_observations: List[ToolObservation] = []
        last_feedback = ""

        while attempts <= self.config.routing.max_worker_retries:
            attempts += 1
            worker, order, result = await self._invoke_worker(task_id, step, feedback_items)
            observations = self._execute_actions(step.capability, result)
            verification = await self._verify(task_id, order, result, observations)
            feedback = self._feedback(result, observations, verification)
            last_feedback = feedback
            self.learning.record_step(
                task_id=task_id,
                step=step,
                worker_id=worker.id,
                model=worker.model,
                result=result,
                observations=observations,
                feedback=feedback,
            )
            final_worker, final_result, final_observations = worker, result, observations
            if not feedback:
                break
            feedback_items.append(feedback)
            allowed = set(self.executor.available_for(step.capability.value))
            mutation_attempted = self.executor.has_side_effects(
                call for call in result.tool_calls if call.name in allowed
            )
            if mutation_attempted or result.status in {WorkerStatus.NEEDS_CLARIFICATION, WorkerStatus.OUT_OF_SCOPE}:
                break

        assert final_worker is not None and final_result is not None
        if last_feedback and final_result.status == WorkerStatus.COMPLETED:
            final_result.status = WorkerStatus.BLOCKED
            final_result.ambiguity = last_feedback
        return StepReport(
            step=step,
            worker_id=final_worker.id,
            result=final_result,
            observations=final_observations,
            attempts=attempts,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def _execute_plan(self, task_id: str, plan: TaskPlan) -> List[StepReport]:
        pending: Dict[str, PlanStep] = {step.id: step for step in plan.steps}
        completed: set[str] = set()
        reports: List[StepReport] = []
        while pending:
            ready = [step for step in pending.values() if set(step.depends_on) <= completed]
            if not ready:
                raise RuntimeError("Plan contains a dependency cycle or an unsatisfied dependency")
            parallel = [step for step in ready if step.parallel_safe]
            batch = (parallel or [ready[0]])[: self.config.routing.max_parallel_steps]
            batch_reports = await asyncio.gather(*(self._run_step(task_id, step) for step in batch))
            for report in batch_reports:
                reports.append(report)
                pending.pop(report.step.id)
                if report.result.status == WorkerStatus.COMPLETED:
                    completed.add(report.step.id)
                else:
                    return reports
        return reports

    async def run(self, goal: str) -> RunReport:
        started = time.perf_counter()
        task_id = str(uuid4())
        self.store.append(CntxEvent(
            type=EventType.USER_INSTRUCTION,
            scope="project",
            priority=Priority.P0,
            source="user",
            task_id=task_id,
            summary=goal,
            payload={"goal": goal},
        ))
        plan, used_fast_path = await self._initial_plan(task_id, goal)
        if plan.user_clarification:
            return RunReport(
                task_id=task_id,
                goal=goal,
                plan=plan,
                status=WorkerStatus.NEEDS_CLARIFICATION,
                clarification=plan.user_clarification,
                total_duration_ms=(time.perf_counter() - started) * 1000,
                model_calls=self.transport.model_calls,
                manager_calls=self.transport.manager_calls,
            )

        reports = await self._execute_plan(task_id, plan)
        failed = next((report for report in reports if report.result.status != WorkerStatus.COMPLETED), None)
        failed_mutated = bool(
            failed
            and self.executor.has_side_effects(
                call
                for call in failed.result.tool_calls
                if call.name in set(self.executor.available_for(failed.step.capability.value))
            )
        )
        can_replan = bool(
            failed
            and self.config.routing.max_manager_replans > 0
            and Capability.MANAGER in self.workers
            and (used_fast_path or not failed_mutated)
        )
        if can_replan:
            failure_context = f"Fast-path worker {failed.worker_id} returned {failed.result.status.value}: {failed.result.summary}"
            plan = await self._manager_plan(task_id, goal, failure_context)
            if plan.user_clarification:
                status = WorkerStatus.NEEDS_CLARIFICATION
                clarification = plan.user_clarification
            else:
                reports.extend(await self._execute_plan(task_id, plan))
                status = WorkerStatus.COMPLETED if reports[-1].result.status == WorkerStatus.COMPLETED else reports[-1].result.status
                clarification = reports[-1].result.ambiguity
        else:
            status = WorkerStatus.COMPLETED if not failed else failed.result.status
            clarification = failed.result.ambiguity if failed else None

        return RunReport(
            task_id=task_id,
            goal=goal,
            plan=plan,
            status=status,
            steps=reports,
            clarification=clarification,
            total_duration_ms=(time.perf_counter() - started) * 1000,
            model_calls=self.transport.model_calls,
            manager_calls=self.transport.manager_calls,
        )
