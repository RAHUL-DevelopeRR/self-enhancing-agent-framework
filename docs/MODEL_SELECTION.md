# V0 Model Selection and Training Decision

Checked on 5 September 2026. Model names and runtime behavior must be pinned in
benchmark manifests because they can change.

## Selected bootstrap models

| DLLM role | V0 model | Why it is a candidate | What remains unproven |
|---|---|---|---|
| Engineering manager | Qwen3-30B-A3B GGUF Q4 | MoE: 30.5B total and 3.3B active; broad reasoning and tool-oriented instruction behavior | Planning quality, load latency and 24 GB operating margin |
| Shell | Qwen2.5-Coder-0.5B-Instruct GGUF Q5 | Compact code-specific foundation; official GGUF; approximately 522 MB at Q5_K_M | Bash/PowerShell safety, exact quoting and error recovery after specialization |
| FileOps | Qwen3-0.6B GGUF Q8 | Compact instruction model with official llama.cpp path; approximately 639 MB | Path semantics, collision handling and stable structured actions |
| Tests | Qwen2.5-Coder-1.5B-Instruct GGUF Q4 | More capacity for test selection and stack-trace interpretation | Whether 0.5/0.6B is sufficient after domain tuning |
| Code | Qwen2.5-Coder-7B-Instruct GGUF Q4 | Code-specific local worker for bounded edits | Real repository benchmark performance |
| Verifier | Qwen3-0.6B GGUF Q8 | Cheap independent inference pass | Calibration and resistance to agreeing with a plausible wrong result |

SmolLM2-360M-Instruct is retained as a serious 300M-class training baseline. Its
official model card reports a 360M variant trained on a broad corpus and designed
for on-device use. It needs conversion or a trusted GGUF build for llama.cpp.

V0 deliberately does not select an obscure “PowerShell specialist” solely from
its name. The Shell benchmark must include exact commands, Windows/POSIX quoting,
stderr repair, refusal and ambiguity. A known compact base is specialized on that
dataset, then promoted only if it passes.

## Decision: specialize, do not pretrain from random initialization yet

Runtime independence means each worker owns a separately versioned checkpoint or
adapter and can operate without the manager. It does not require throwing away a
base model's language and syntax pretraining.

The first experiment is:

1. Qwen2.5-Coder-0.5B base/instruct checkpoint;
2. domain-adaptive Shell/Bash/PowerShell corpus;
3. structured manager-work-order to worker-result distillation;
4. negative examples for destructive commands and missing context;
5. QLoRA/LoRA specialist tuning;
6. GGUF conversion and Q5 benchmark;
7. independent deployment as `shell-worker-v1`.

Repeat separately for FileOps. Do not combine their datasets until an ablation
shows that a shared model is better than independent specialists.

## Primary references

- [llama.cpp HTTP server and multi-model router](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Qwen3-30B-A3B official GGUF model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)
- [Qwen2.5-Coder-0.5B-Instruct official GGUF model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF)
- [Qwen3-0.6B official GGUF model card](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
- [SmolLM2-360M-Instruct official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [Google Research TurboQuant overview](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
