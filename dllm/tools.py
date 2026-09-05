from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import ToolPolicyConfig
from .schemas import ToolCall, ToolObservation


class ToolPolicyError(RuntimeError):
    pass


class SafeToolExecutor:
    """Deterministic action plane. Models propose; this class validates and acts."""

    TOOL_CAPABILITIES = {
        "list_files": "fileops",
        "search_text": "fileops",
        "read_file": "fileops",
        "write_file": "fileops",
        "run_command": "shell",
        "run_tests": "test",
    }
    SIDE_EFFECTING_TOOLS = {"write_file", "run_command"}

    @classmethod
    def has_side_effects(cls, calls: Iterable[ToolCall]) -> bool:
        return any(call.name in cls.SIDE_EFFECTING_TOOLS for call in calls)

    def __init__(self, project_root: str | Path, policy: ToolPolicyConfig):
        self.project_root = Path(project_root).resolve()
        self.policy = policy

    def available_for(self, capability: str) -> List[str]:
        if capability == "code":
            return ["read_file", "search_text", "write_file"]
        return [name for name, owner in self.TOOL_CAPABILITIES.items() if owner == capability]

    def _path(self, raw: str) -> Path:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve()
        if self.policy.project_root_only and resolved != self.project_root and self.project_root not in resolved.parents:
            raise ToolPolicyError(f"Path escapes project root: {raw}")
        return resolved

    def execute(self, call: ToolCall) -> ToolObservation:
        started = time.perf_counter()
        try:
            handler = getattr(self, f"_tool_{call.name}", None)
            if handler is None:
                raise ToolPolicyError(f"Unknown tool: {call.name}")
            data = handler(call.arguments)
            summary = str(data.pop("summary", f"{call.name} completed"))
            return ToolObservation(
                tool=call.name,
                ok=True,
                summary=summary,
                data=data,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolObservation(
                tool=call.name,
                ok=False,
                summary=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

    def _tool_list_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        root = self._path(str(args.get("path", ".")))
        pattern = str(args.get("pattern", "*"))
        recursive = bool(args.get("recursive", True))
        iterator: Iterable[Path] = root.rglob(pattern) if recursive else root.glob(pattern)
        paths = [str(path.relative_to(self.project_root)) for path in iterator if path.is_file()]
        paths = paths[: int(args.get("limit", 500))]
        return {"summary": f"Found {len(paths)} files", "paths": paths}

    def _tool_search_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        root = self._path(str(args.get("path", ".")))
        query = str(args["query"])
        pattern = str(args.get("pattern", "*"))
        case_sensitive = bool(args.get("case_sensitive", False))
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(re.escape(query), flags)
        matches = []
        for path in root.rglob(pattern):
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        matches.append({"path": str(path.relative_to(self.project_root)), "line": number, "text": line[:300]})
                        if len(matches) >= int(args.get("limit", 100)):
                            return {"summary": f"Found {len(matches)} matches", "matches": matches}
            except (UnicodeDecodeError, OSError):
                continue
        return {"summary": f"Found {len(matches)} matches", "matches": matches}

    def _tool_read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = self._path(str(args["path"]))
        max_chars = min(int(args.get("max_chars", 12000)), self.policy.max_output_chars)
        text = path.read_text(encoding="utf-8")
        return {
            "summary": f"Read {path.relative_to(self.project_root)}",
            "path": str(path.relative_to(self.project_root)),
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    def _tool_write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.policy.allow_writes:
            raise ToolPolicyError("Writes are disabled; start with --allow-write")
        path = self._path(str(args["path"]))
        content = str(args["content"])
        expected = args.get("expected_sha256")
        if path.exists() and not expected:
            raise ToolPolicyError("Overwriting an existing file requires expected_sha256")
        if path.exists() and expected:
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            if current != expected:
                raise ToolPolicyError("File changed since observation; hash precondition failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"summary": f"Wrote {path.relative_to(self.project_root)} atomically", "path": str(path.relative_to(self.project_root))}

    def _validated_command(self, args: Dict[str, Any]) -> List[str]:
        if not self.policy.allow_shell:
            raise ToolPolicyError("Shell execution is disabled; start with --allow-shell")
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise ToolPolicyError("Commands must be supplied as a non-empty argv string array")
        joined = " ".join(argv).lower()
        if any(fragment.lower() in joined for fragment in self.policy.blocked_command_fragments):
            raise ToolPolicyError("Command matched a blocked destructive pattern")
        return argv

    def _run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        argv = self._validated_command(args)
        completed = subprocess.run(
            argv,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=self.policy.command_timeout_seconds,
            shell=False,
        )
        stdout = completed.stdout[-self.policy.max_output_chars:]
        stderr = completed.stderr[-self.policy.max_output_chars:]
        return {
            "summary": f"Command exited with code {completed.returncode}",
            "argv": argv,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _tool_run_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(args)

    def _tool_run_tests(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(args)
