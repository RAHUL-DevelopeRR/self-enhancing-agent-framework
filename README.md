# Self-Enhancing Autonomous Agentic Framework

## DLLM V0 prototype

This repository now includes a runnable V0 of the **Decentralized/Distributed
Specialist LLM Runtime**: a medium engineering-manager model delegates bounded
work to independent Shell, FileOps, Code, Testing, Verification and Web/RAG
workers served through llama.cpp. Typed tool boundaries, append-only `.cntx`
project documentation, phase-gated model activation and the existing episodic
self-enhancement loop are integrated.

Start with [the V0 architecture](docs/DLLM_V0_ARCHITECTURE.md), then read the
[boundary and ambiguity contract](docs/DLLM_BOUNDARIES.md) and the
[model-selection decision](docs/MODEL_SELECTION.md).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dllm.txt
winget install llama.cpp
./scripts/start_dllm_llama.ps1
.\.venv\Scripts\python.exe -m dllm init
.\.venv\Scripts\python.exe -m dllm doctor
.\.venv\Scripts\python.exe -m dllm run "List files containing AgenticLoop"
```

Writes and shell execution are disabled unless explicitly enabled. The selected
GGUF models are bootstrap candidates for V0; [specialist training](training/README.md)
begins after traces establish measurable failure cases and accepted corrections.

An independent, model-agnostic agentic framework designed to scrape lead datasets, conduct personalized omnichannel outreach (WhatsApp, Instagram, Zoho Mail), automatically generate responsive websites upon client agreement, and showcase client transformations on social media.

Built upon a **two-tier decoupled architecture**:
1. **Intelligent API Broker**: Dynamically routes and quota-manages across free and low-cost AI providers (Groq, Cerebras, Cloudflare Workers AI, OpenRouter, Gemini).
2. **Self-Enhancing Client Core**: Accumulates persistent intelligence across tasks using zero-token local RAG, episodic memory, a Draft-Critic-Refine loop, and automated lesson harvesting.

---

## Architecture Diagram

```
                              FREE / LOW-COST PROVIDER POOL
                  ┌──────────┬───────────┬────────────┬──────────┬────────────┐
                  │   Groq   │ Cerebras  │ Cloudflare │  Gemini  │ OpenRouter │
                  └────┬─────┴─────┬─────┴─────┬──────┴────┬─────┴──────┬─────┘
                       │           │           │           │            │
                       └───────────┼───────────┼───────────┼────────────┘
                                   ▼           ▼           ▼
                         ┌───────────────────────────────────────┐
                         │              API BROKER               │
                         │  • Rate-limit header quota tracking   │
                         │  • Intelligent capability router      │
                         │  • Dynamic route scoring & failover   │
                         └───────────────────┬───────────────────┘
                                             │ Unified POST /v1/chat
                                             ▼
                         ┌───────────────────────────────────────┐
                         │         SELF-ENHANCING CLIENT         │
                         │  • Zero-token Local Memory & Lessons  │
                         │  • Dynamic Context Composer           │
                         │  • Draft -> Critic -> Refine Loop     │
                         │  • Experience & Lesson Harvester      │
                         └───────────────────┬───────────────────┘
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
              [STAGE 1: SCRAPE]      [STAGE 2: OUTREACH]    [STAGE 3: DELIVER]
              Apify Cloud Actors     Composio (Zoho,        Automated Site Builder
              (Google Maps, Bios)    WhatsApp, Instagram)   & Social Showcase
```

---

## 4 Layers of Self-Enhancement

1. **Response Learning**: Tasks run through a multi-model Draft -> Critic -> Refine loop (Mixture of Agents, capped at 2 iterations) to guarantee high quality before dispatch.
2. **Memory Learning**: Every run is logged in an SQLite `episodes` database with critic scores, prompts, and failure points.
3. **Policy Learning**: The `ExperienceHarvester` analyzes recurring failures and crystallizes them into compact, trigger-based rules (`lessons` table). These rules are dynamically injected into future prompts using zero-token local vector similarity.
4. **Routing Learning**: The Model Router scores providers based on:
   $$\text{Score} = (\text{Quality} \times 0.35) + (\text{Quota Left} \times 0.20) + (\text{Reliability} \times 0.20) + (\text{Capability} \times 0.15) + (\text{Speed} \times 0.10)$$

---

## Model Context Protocol (MCP) Configuration

The framework comes preconfigured with Model Context Protocol definitions in [`mcp/mcp_servers.json`](mcp/mcp_servers.json):

* **Apify**: Scrapes Google Maps business leads, Instagram bios, and website emails with rotating proxies.
* **Composio**: Connects Zoho Mail, Instagram Graph API, and WhatsApp Business API.
* **WhatsApp**: Local QR-code web-bridge (`whatsapp-mcp`) for personal chat monitoring.
* **Browser**: Headless browser automation via Puppeteer.

---

## Uploading GitHub Secrets via `gh cli`

To push your API tokens (`APIFY_TOKEN`, `COMPOSIO_API_KEY`, etc.) as encrypted GitHub Actions secrets:

### 1. Authenticate with GitHub CLI:
```bash
gh auth login
```

### 2. Run the automated secrets uploader:
* **On Windows (PowerShell)**:
  ```powershell
  .\scripts\setup_secrets.ps1
  ```
* **On Linux / macOS (Bash)**:
  ```bash
  chmod +x ./scripts/setup_secrets.sh
  ./scripts/setup_secrets.sh
  ```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```

### 3. Run the Autonomous Pipeline
```bash
python main.py
```
