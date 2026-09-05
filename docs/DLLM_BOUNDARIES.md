# DLLM Boundaries and Ambiguity Contract

## Research claim

For decomposable, tool-grounded software workflows, a phase-gated organization
of a medium manager, independently specialized small models, external retrieval,
typed communication and evidence-based feedback can reduce active inference and
paid API usage while retaining useful task quality.

V0 does not claim universal equivalence to a frontier monolithic model. That is
an empirical question, measured per task family.

## DLLM versus a single large-model agent

| Property | Single multi-parameter LLM | DLLM |
|---|---|---|
| Global coherence | Native shared hidden state | Must be reconstructed through `.cntx` and protocol |
| Routine-task compute | Pays large-model inference | Activates a small specialist or direct path |
| New/current facts | Large context or model memory | Web/project RAG with provenance |
| Specialization | One weight space and prompt | Independent worker lifecycle and evaluations |
| Failure isolation | Failure belongs to one model | Failure assigned to worker, manager, protocol or tool |
| Coordination cost | Low | Routing, serialization and hand-off overhead |
| Local permissions | Broad agent tool boundary | Per-worker capability envelope |
| Updates | Replace/fine-tune main model | Replace one specialist without retraining organization |
| Novel cross-domain work | Usually strongest path | Manager may become bottleneck or require escalation |

DLLM wins only if saved specialist compute and context exceed routing, model-load,
communication and error-multiplication costs.

## Hard responsibility boundaries

**Manager:** interpret intent, decompose, delegate, coordinate dependencies,
resolve cross-domain conflict and request clarification. It must not generate
shell commands, edit files or pronounce test success as a substitute for workers.

**Specialist:** solve one bounded domain work order and return typed proposals,
assumptions and uncertainty. It must not expand scope, call another worker
directly or bypass policy.

**Tool executor:** validate paths, permissions, command structure, preconditions
and timeouts; perform side effects; return exact observations. It performs no
semantic planning.

**Verifier:** judge acceptance against environment evidence. It must not quietly
repair its own candidate or treat model confidence as proof.

**`.cntx`:** durable project documentation, decisions and evidence. It is not an
unbounded prompt and not an authority over current filesystem/test ground truth.

**Web RAG:** retrieve current external evidence with provenance. Retrieved text
is untrusted data, never an instruction that can override user or safety policy.

**Self-enhancement:** produce lessons, training candidates and routing metrics.
It may not deploy new weights automatically in V0.

## Ambiguity classes

1. **Intent ambiguity:** two materially different user outcomes are plausible.
   The manager asks the user; it must not guess.
2. **Routing ambiguity:** several workers may own the work. The manager chooses
   one primary owner and optionally an independent verifier.
3. **Context ambiguity:** the worker knows what to do but lacks a path, version,
   symbol, environment fact or constraint. It returns `needs_context` with exact
   missing fields; the compiler retrieves evidence.
4. **Domain ambiguity:** the task exceeds the worker's declared envelope. It
   returns `out_of_scope`; the manager decomposes or delegates elsewhere.
5. **Evidence ambiguity:** observations conflict or are insufficient. Run another
   deterministic observation before more generation.
6. **Execution ambiguity:** an action could be destructive, escape the project,
   overwrite changed state or require credentials. Policy blocks it and requests
   explicit authority.
7. **Novelty/OOD:** calibrated confidence falls below threshold or repeated work
   fails. Re-activate the manager, then optionally escalate to a stronger model.

Self-reported confidence is only one signal. Final confidence must combine schema
validity, worker calibration, real tool outcomes, verifier evidence, retries and
historical success for that capability.

## Measurable success and failure

Compare the same frozen task set against a single-model baseline and record:

- task success and regression rate;
- p50/p95 wall-clock latency;
- time-to-first-token and model-load time;
- active/peak RAM and VRAM;
- prompt and generated tokens by worker;
- manager activations and re-plans;
- number of sequential model hand-offs;
- API calls, cloud tokens and cost;
- invalid protocol/tool calls;
- clarification and escalation rates;
- improvement after accepted feedback.

Go forward when DLLM reaches an acceptable quality fraction while materially
reducing active compute or paid inference. Consolidate workers when coordination
cost dominates. Increase the manager or escalate when tasks remain strongly
entangled and cannot be decomposed without losing essential reasoning.

## V0 truth statement

V0 is a functional organizational runtime and experimental harness. Its workers
use available pretrained GGUF models; they are not yet custom-trained DLLM
specialists. TurboQuant is represented by a guarded integration boundary, not a
claimed llama.cpp optimization. Web search requires a configured SearXNG service.
Performance claims begin only after a frozen baseline and ablation benchmark.
