from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from client.memory import EpisodicMemory

from .compression import compression_status
from .config import load_config
from .cntx import CntxStore
from .orchestrator import DLLMOrchestrator
from .runtime import LlamaCppTransport


def _config_path(raw: str) -> Path:
    return Path(raw).resolve()


def initialize(project_root: Path, config_path: Path) -> None:
    config = load_config(config_path)
    cntx_path = Path(config.context.path)
    if not cntx_path.is_absolute():
        cntx_path = project_root / cntx_path
    store = CntxStore(cntx_path)
    existing_documents = {
        event.payload.get("title"): event.payload.get("sha256") or hashlib.sha256(
            str(event.payload.get("content", "")).encode("utf-8")
        ).hexdigest()
        for event in store.iter_events()
        if event.type.value == "DOCUMENT"
    }
    docs = [
        project_root / "docs" / "DLLM_V0_ARCHITECTURE.md",
        project_root / "docs" / "DLLM_BOUNDARIES.md",
        project_root / "docs" / "MODEL_SELECTION.md",
    ]
    added = 0
    for document in docs:
        if document.exists():
            content = document.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if existing_documents.get(document.stem) == content_hash:
                continue
            store.document(document.stem, content)
            added += 1
    print(f"Project documentation ready: {cntx_path} ({added} new documents)")


async def doctor(project_root: Path, config_path: Path) -> int:
    config = load_config(config_path)
    cntx_path = project_root / config.context.path
    store = CntxStore(cntx_path)
    errors = store.verify()
    transport = LlamaCppTransport()
    enabled = [worker for worker in config.workers if worker.enabled]
    health = await asyncio.gather(*(transport.health(worker) for worker in enabled))
    checks = [(worker.id, worker_health) for worker, worker_health in zip(enabled, health)]
    await transport.close()
    status = compression_status(config.context.semantic_compaction, config.context.vector_backend)
    print(json.dumps({
        "cntx": {"path": str(cntx_path), "integrity_errors": errors},
        "workers": {name: healthy for name, healthy in checks},
        "compression": status.__dict__,
    }, indent=2))
    return 0 if not errors and all(healthy for _, healthy in checks) else 1


async def execute(args) -> int:
    project_root = Path(args.project_root).resolve()
    config = load_config(_config_path(args.config))
    if args.allow_shell:
        config.tools.allow_shell = True
    if args.allow_write:
        config.tools.allow_writes = True
    memory_path = project_root / ".dllm" / "agent_memory.db"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory = EpisodicMemory(str(memory_path))
    transport = LlamaCppTransport()
    organization = DLLMOrchestrator(config, project_root, transport=transport, episodic_memory=memory)
    try:
        report = await organization.run(args.goal)
    finally:
        await transport.close()
    print(report.model_dump_json(indent=2))
    return 0 if report.status.value == "completed" else 2


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dllm", description="DLLM V0 local model organization")
    parser.add_argument("--config", default="configs/dllm_v0.json")
    parser.add_argument("--project-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Initialize append-only project documentation")
    subparsers.add_parser("doctor", help="Check .cntx integrity, endpoints, and compression status")
    run_parser = subparsers.add_parser("run", help="Delegate a goal to the DLLM organization")
    run_parser.add_argument("goal")
    run_parser.add_argument("--allow-shell", action="store_true")
    run_parser.add_argument("--allow-write", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = _config_path(args.config)
    if args.command == "init":
        initialize(root, config_path)
    elif args.command == "doctor":
        raise SystemExit(asyncio.run(doctor(root, config_path)))
    else:
        raise SystemExit(asyncio.run(execute(args)))


if __name__ == "__main__":
    main()
