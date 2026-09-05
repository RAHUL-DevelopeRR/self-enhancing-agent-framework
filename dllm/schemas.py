from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Capability(str, Enum):
    MANAGER = "manager"
    SHELL = "shell"
    FILEOPS = "fileops"
    CODE = "code"
    TEST = "test"
    VERIFY = "verify"
    WEB_RESEARCH = "web_research"


class WorkerStatus(str, Enum):
    COMPLETED = "completed"
    NEEDS_CONTEXT = "needs_context"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"
    BLOCKED = "blocked"
    FAILED = "failed"


class RiskClass(str, Enum):
    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class EventType(str, Enum):
    DOCUMENT = "DOCUMENT"
    USER_INSTRUCTION = "USER_INSTRUCTION"
    DECISION = "DECISION"
    WORK_ORDER = "WORK_ORDER"
    WORKER_RESULT = "WORKER_RESULT"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    TEST_RESULT = "TEST_RESULT"
    PATCH = "PATCH"
    ERROR = "ERROR"
    FEEDBACK = "FEEDBACK"
    LESSON = "LESSON"
    WEB_EVIDENCE = "WEB_EVIDENCE"
    SUMMARY = "SUMMARY"
    ESCALATION = "ESCALATION"
    CHECKPOINT = "CHECKPOINT"


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    risk: RiskClass = RiskClass.READ_ONLY
    rationale: str = ""


class ToolObservation(BaseModel):
    tool: str
    ok: bool
    summary: str
    data: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class PlanStep(BaseModel):
    id: str
    capability: Capability
    objective: str
    depends_on: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    parallel_safe: bool = False


class TaskPlan(BaseModel):
    goal: str
    summary: str
    steps: List[PlanStep]
    assumptions: List[str] = Field(default_factory=list)
    user_clarification: Optional[str] = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> "TaskPlan":
        identifiers = {step.id for step in self.steps}
        if len(identifiers) != len(self.steps):
            raise ValueError("Plan step IDs must be unique")
        for step in self.steps:
            unknown = set(step.depends_on) - identifiers
            if unknown:
                raise ValueError(f"Step {step.id} has unknown dependencies: {sorted(unknown)}")
            if step.id in step.depends_on:
                raise ValueError(f"Step {step.id} cannot depend on itself")
        return self


class WorkOrder(BaseModel):
    task_id: str
    step_id: str
    capability: Capability
    objective: str
    constraints: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    feedback: List[str] = Field(default_factory=list)


class WorkerResult(BaseModel):
    step_id: str
    status: WorkerStatus
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    ambiguity: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list)
    recommended_capability: Optional[Capability] = None


class CntxEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: EventType
    scope: str = "project"
    priority: Priority = Priority.P2
    source: str
    task_id: Optional[str] = None
    parent_id: Optional[str] = None
    summary: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    artifact_refs: List[str] = Field(default_factory=list)
    provenance: List[str] = Field(default_factory=list)
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None


class ContextPackage(BaseModel):
    task_id: str
    role: Capability
    goal: str
    constraints: List[str] = Field(default_factory=list)
    current_state: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge: List[Dict[str, Any]] = Field(default_factory=list)
    lessons: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: List[str] = Field(default_factory=list)
    approximate_tokens: int = 0


class StepReport(BaseModel):
    step: PlanStep
    worker_id: str
    result: WorkerResult
    observations: List[ToolObservation] = Field(default_factory=list)
    attempts: int = 1
    duration_ms: float = 0.0


class RunReport(BaseModel):
    task_id: str
    goal: str
    plan: TaskPlan
    status: WorkerStatus
    steps: List[StepReport] = Field(default_factory=list)
    clarification: Optional[str] = None
    total_duration_ms: float = 0.0
    model_calls: int = 0
    manager_calls: int = 0
