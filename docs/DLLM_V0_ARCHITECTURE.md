# DLLM V0 Architecture

## Definition

DLLM is a model organization. A medium general reasoner acts as engineering
manager; independent specialist language models act as workers; deterministic
tools operate the computer; `.cntx` is their shared project documentation and
evidence ledger; the self-enhancing loop turns measured outcomes into role-local
feedback and future training examples.

It is not one model receiving several role prompts, and it is not a group chat
between generic agents. A production specialist has its own weights or adapter,
dataset, capability envelope, evaluation suite, version and deployment identity.

## Runtime organization

```text
User / IDE
    |
    v
Engineering manager (medium model or MoE; planning and ambiguity only)
    |
    +-- Shell worker ------> validated argv ------> process executor
    +-- FileOps worker ----> typed file action ---> filesystem executor
    +-- Code worker -------> bounded change ------> atomic writer
    +-- Test worker -------> test selection ------> test runner
    +-- Web worker --------> evidence query ------> Web RAG
    |
    v
Independent verifier -> feedback -> worker retry or manager re-plan
    |
    v
Append outcome, evidence and lesson to .cntx + existing episodic memory
```

The manager is activated for decomposition, cross-domain reasoning, conflict and
material ambiguity. Explicit routine requests can use a direct specialist fast
path. Specialists never directly mutate the environment: they submit typed
actions to a policy-enforced deterministic executor.

## `.cntx`: shared project documentation

`.dllm/project.cntx` is append-only JSONL. It stores human-authored project
documentation and runtime evidence in the same durable namespace:

- `DOCUMENT`: architecture, conventions, subsystem documentation;
- `USER_INSTRUCTION`: goals, corrections and immutable constraints;
- `DECISION`: accepted technical decisions and rationale;
- `WORK_ORDER` and `WORKER_RESULT`: delegation history;
- `OBSERVATION`, `TEST_RESULT` and `ERROR`: real environment evidence;
- `FEEDBACK` and `LESSON`: self-enhancement material;
- `WEB_EVIDENCE`: retrieved content with URL and content hash.

Events are hash-linked. The context compiler retrieves only events relevant to
the current worker and token budget. P0 user/safety constraints and P1 project
decisions receive priority. Raw events are not destroyed after summarization.

Rule: append forever; never inject forever.

## Inter-model contract

The manager emits a typed task DAG. Each worker receives one `WorkOrder` with:

- objective and acceptance criteria;
- capability-specific tools;
- constraints and previous feedback;
- a bounded `.cntx` context package with provenance.

Each worker returns a `WorkerResult` with status, confidence, typed tool calls,
assumptions, missing information, ambiguity and evidence references. Models do
not converse in unconstrained prose.

## Self-enhancement

V0 performs safe online organizational learning, not autonomous weight mutation.

1. Every specialist result is scored against real tool observations.
2. Failed execution or verifier rejection becomes targeted worker feedback.
3. Read-only work may be retried once with that feedback.
4. Episodes are also recorded in the existing SQLite memory.
5. A later successful result can be exported as a role-specific fine-tuning pair.
6. Weight updates happen offline and must pass the specialist evaluation gate.

Shell failures improve the Shell worker; FileOps failures improve FileOps;
incorrect decomposition improves the manager dataset. This prevents unrelated
lessons from polluting every worker.

## Current model bootstrap

V0 uses available GGUF checkpoints to validate the organization before paying
the cost of specialist training:

- manager: `Qwen3-30B-A3B` Q4, an MoE with 3.3B active parameters;
- Shell: `Qwen2.5-Coder-0.5B-Instruct` Q5;
- FileOps: `Qwen3-0.6B` Q8;
- testing: `Qwen2.5-Coder-1.5B-Instruct` Q4;
- coding: `Qwen2.5-Coder-7B-Instruct` Q4;
- verifier: `Qwen3-0.6B` Q8.

These are bootstrap candidates, not evidence that they are already optimized
specialists. The first training target is the 0.5B Shell model, followed by the
0.6B FileOps model. The registry is configuration, so each checkpoint can be
replaced without changing the protocol.

## Latency and memory strategy

V0 uses llama.cpp router mode. The request's model identity autoloads the relevant
GGUF; idle models sleep and release weight/KV memory. The following controls are
required:

1. Direct-route explicit routine work without a manager generation.
2. Call the manager once to emit a complete bounded DAG.
3. Keep work orders and outputs schema-constrained and short.
4. Inject role-specific `.cntx`, never the full session.
5. Run independent plan steps concurrently, up to a configured limit.
6. Keep tool output out of prompts; store it once and pass references/summaries.
7. Use Q4/Q5 weights and llama.cpp KV quantization.
8. Sleep the large manager after its planning phase.
9. Re-activate the manager only for ambiguity, conflicting evidence or re-plan.

This trades large-model loading latency against peak memory. A second deployment
profile may keep the manager resident on another GPU or private API while tiny
workers remain local; the DMP contracts do not change.

## TurboQuant boundary

TurboQuant is numerical vector/KV-cache compression. It does not compress raw
`.cntx` prose and is not a replacement for semantic context compilation.

V0 therefore has three separate controls:

- context compiler: reduces injected tokens;
- llama.cpp Q4/Q5 and quantized KV: reduces currently supported runtime memory;
- optional TurboQuant vector backend: an experimental integration seam for a
  future embedding index or runtime that has a validated TurboQuant kernel.

The `doctor` command reports whether an optional `turboquant` module is present,
but V0 does not claim that upstream llama.cpp uses TurboQuant. It must be enabled
only after an implementation is pinned, audited and benchmarked independently.

## Run V0

```powershell
winget install llama.cpp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dllm.txt
# Terminal 1: keep the local model router running
./scripts/start_dllm_llama.ps1
# Terminal 2
.\.venv\Scripts\python.exe -m dllm init
.\.venv\Scripts\python.exe -m dllm doctor
.\.venv\Scripts\python.exe -m dllm run "List files containing AgenticLoop"
```

Shell and writes are deny-by-default:

```powershell
python -m dllm run "Run tests for the DLLM package" --allow-shell
python -m dllm run "Implement the accepted change" --allow-shell --allow-write
```

Enable Web RAG by supplying a SearXNG endpoint in `configs/dllm_v0.json`. Retrieved
pages are bounded, domain-filterable, content-hashed and written to `.cntx` with
their source URL before a web specialist interprets them.
