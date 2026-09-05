from __future__ import annotations

import argparse
import json
from pathlib import Path

from dllm.cntx import CntxStore
from dllm.schemas import EventType


def export_examples(source: Path, destination: Path, capability: str | None) -> int:
    store = CntxStore(source)
    feedback = {}
    examples = []
    for event in store.iter_events():
        if event.type == EventType.FEEDBACK:
            key = (event.task_id, event.payload.get("step_id"))
            feedback[key] = event.summary
        elif event.type == EventType.WORKER_RESULT:
            step = event.payload.get("step", {})
            result = event.payload.get("result", {})
            key = (event.task_id, step.get("id"))
            if not event.payload.get("success") or key not in feedback:
                continue
            if capability and step.get("capability") != capability:
                continue
            examples.append({
                "capability": step.get("capability"),
                "instruction": step.get("objective"),
                "critic_feedback": feedback[key],
                "response": result,
                "provenance": [f"cntx:{event.id}"],
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".dllm/project.cntx")
    parser.add_argument("--output", default="training/derived/accepted_examples.jsonl")
    parser.add_argument("--capability", choices=["shell", "fileops", "code", "test", "verify"])
    args = parser.parse_args()
    count = export_examples(Path(args.source), Path(args.output), args.capability)
    print(f"Exported {count} accepted specialist examples")
