import tempfile
import unittest
from pathlib import Path

from dllm.config import DLLMConfig, WorkerConfig
from dllm.cntx import CntxStore, ContextCompiler
from dllm.orchestrator import DLLMOrchestrator
from dllm.schemas import (
    Capability,
    CntxEvent,
    EventType,
    Priority,
    TaskPlan,
    ToolCall,
    WorkerResult,
    WorkerStatus,
)


class FakeTransport:
    def __init__(self):
        self.model_calls = 0
        self.manager_calls = 0

    async def generate(self, worker, payload, response_model):
        self.model_calls += 1
        if worker.capability == Capability.MANAGER:
            self.manager_calls += 1
            return TaskPlan.model_validate({
                "goal": payload["goal"],
                "summary": "delegate",
                "steps": [{"id": "s1", "capability": "fileops", "objective": "list files"}],
            })
        order = payload["work_order"]
        return WorkerResult(
            step_id=order["step_id"],
            status=WorkerStatus.COMPLETED,
            summary="prepared bounded file listing",
            confidence=0.95,
            tool_calls=[ToolCall(name="list_files", arguments={"path": ".", "pattern": "*.py"})],
        )


class RetryTransport(FakeTransport):
    async def generate(self, worker, payload, response_model):
        self.model_calls += 1
        order = payload["work_order"]
        if self.model_calls == 1:
            return WorkerResult(
                step_id=order["step_id"],
                status=WorkerStatus.COMPLETED,
                summary="first attempt used wrong capability",
                confidence=0.9,
                tool_calls=[ToolCall(name="run_command", arguments={"argv": ["git", "status"]})],
            )
        return WorkerResult(
            step_id=order["step_id"],
            status=WorkerStatus.COMPLETED,
            summary="corrected after manager feedback",
            confidence=0.95,
            tool_calls=[ToolCall(name="list_files", arguments={"path": ".", "pattern": "*.py"})],
        )


class AmbiguityTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.worker_calls = 0

    async def generate(self, worker, payload, response_model):
        self.model_calls += 1
        if worker.capability == Capability.MANAGER:
            self.manager_calls += 1
            return TaskPlan.model_validate({
                "goal": payload["goal"],
                "summary": "manager resolved the ownership ambiguity",
                "steps": [{"id": "resolved-1", "capability": "fileops", "objective": "list Python files"}],
            })
        self.worker_calls += 1
        order = payload["work_order"]
        if self.worker_calls == 1:
            return WorkerResult(
                step_id=order["step_id"],
                status=WorkerStatus.OUT_OF_SCOPE,
                summary="request needs manager decomposition",
                confidence=0.3,
                ambiguity="ownership is unclear",
            )
        return WorkerResult(
            step_id=order["step_id"],
            status=WorkerStatus.COMPLETED,
            summary="resolved work order completed",
            confidence=0.95,
            tool_calls=[ToolCall(name="list_files", arguments={"path": ".", "pattern": "*.py"})],
        )


def make_config() -> DLLMConfig:
    return DLLMConfig(workers=[
        WorkerConfig(id="manager", capability="manager", model="manager", system_prompt="manager"),
        WorkerConfig(id="files", capability="fileops", model="files", system_prompt="files"),
    ])


class CntxTests(unittest.TestCase):
    def test_hash_chain_and_role_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CntxStore(Path(directory) / "project.cntx")
            store.document("Architecture", "Workers obey capability boundaries")
            store.append(CntxEvent(
                type=EventType.OBSERVATION,
                source="tool:test",
                priority=Priority.P2,
                summary="Found file operations implementation",
                payload={"path": "dllm/tools.py"},
            ))
            self.assertEqual([], store.verify())
            package = ContextCompiler(store).compile(
                task_id="t1", role=Capability.FILEOPS, goal="find file operations"
            )
            self.assertTrue(package.knowledge)
            self.assertTrue(package.current_state)
            self.assertTrue(all(ref.startswith("cntx:") for ref in package.provenance))

    def test_project_root_boundary(self):
        from dllm.config import ToolPolicyConfig
        from dllm.schemas import ToolCall
        from dllm.tools import SafeToolExecutor

        with tempfile.TemporaryDirectory() as directory:
            executor = SafeToolExecutor(directory, ToolPolicyConfig())
            observation = executor.execute(ToolCall(name="read_file", arguments={"path": "../outside.txt"}))
            self.assertFalse(observation.ok)
            self.assertIn("escapes project root", observation.summary)

    def test_existing_file_write_requires_observed_hash(self):
        from dllm.config import ToolPolicyConfig
        from dllm.schemas import RiskClass, ToolCall
        from dllm.tools import SafeToolExecutor

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.txt"
            path.write_text("original", encoding="utf-8")
            executor = SafeToolExecutor(directory, ToolPolicyConfig(allow_writes=True))
            observation = executor.execute(ToolCall(
                name="write_file",
                arguments={"path": "existing.txt", "content": "replacement"},
                risk=RiskClass.MEDIUM,
            ))
            self.assertFalse(observation.ok)
            self.assertEqual("original", path.read_text(encoding="utf-8"))

    def test_web_rag_blocks_local_targets(self):
        from dllm.config import WebRagConfig
        from dllm.web_rag import SearxngWebRag

        with tempfile.TemporaryDirectory() as directory:
            rag = SearxngWebRag(WebRagConfig(enabled=True), CntxStore(Path(directory) / "p.cntx"))
            self.assertFalse(rag._allowed("http://127.0.0.1/private"))
            self.assertFalse(rag._allowed("file:///etc/passwd"))
            self.assertTrue(rag._allowed("https://example.com/reference"))

    def test_long_document_is_compacted_to_role_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CntxStore(Path(directory) / "project.cntx")
            content = ("Authentication uses signed tokens. " * 300) + ("Unrelated deployment detail. " * 300)
            store.document("Authentication architecture", content)
            package = ContextCompiler(store, semantic_compaction=True).compile(
                task_id="t2",
                role=Capability.CODE,
                goal="change authentication token validation",
                budget_tokens=250,
            )
            self.assertLessEqual(package.approximate_tokens, 300)
            self.assertTrue(package.knowledge[0]["compacted"])
            self.assertIn("Authentication", package.knowledge[0]["payload"]["content"])


class OrganizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_path_skips_manager_and_uses_real_file_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("print('ok')", encoding="utf-8")
            config = make_config()
            config.context.path = ".dllm/project.cntx"
            transport = FakeTransport()
            organization = DLLMOrchestrator(config, root, transport=transport)
            report = await organization.run("List files in this project")
            self.assertEqual(WorkerStatus.COMPLETED, report.status)
            self.assertEqual(0, report.manager_calls)
            self.assertEqual(["sample.py"], report.steps[0].observations[0].data["paths"])
            self.assertEqual([], organization.store.verify())

    async def test_manager_handles_nontrivial_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FakeTransport()
            organization = DLLMOrchestrator(make_config(), directory, transport=transport)
            report = await organization.run("Investigate the architecture")
            self.assertEqual(1, report.manager_calls)
            self.assertEqual(Capability.FILEOPS, report.plan.steps[0].capability)

    async def test_feedback_retries_read_only_worker_and_preserves_attempts(self):
        from client.memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("print('ok')", encoding="utf-8")
            transport = RetryTransport()
            memory = EpisodicMemory(str(root / "memory.db"))
            organization = DLLMOrchestrator(make_config(), root, transport=transport, episodic_memory=memory)
            report = await organization.run("List files in this project")
            self.assertEqual(WorkerStatus.COMPLETED, report.status)
            self.assertEqual(2, report.steps[0].attempts)
            feedback = [event for event in organization.store.iter_events() if event.type == EventType.FEEDBACK]
            self.assertEqual(1, len(feedback))

    async def test_fast_path_ambiguity_returns_to_manager(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text("print('ok')", encoding="utf-8")
            transport = AmbiguityTransport()
            organization = DLLMOrchestrator(make_config(), root, transport=transport)
            report = await organization.run("List files in this project")
            self.assertEqual(WorkerStatus.COMPLETED, report.status)
            self.assertEqual(1, report.manager_calls)
            self.assertEqual("resolved-1", report.plan.steps[0].id)


if __name__ == "__main__":
    unittest.main()
