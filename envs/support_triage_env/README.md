---
title: Support Triage Environment Server
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Support Triage Environment

`support_triage_env` is a deterministic OpenEnv benchmark for a real workflow humans do every day: triaging customer support tickets. The agent must inspect a ticket, classify it, set a priority, route it to the correct queue, draft a reply, and resolve or escalate the case.

The benchmark is fully offline. Tasks, graders, and rewards are all in-repo, which keeps scores reproducible across local runs, Docker, and Hugging Face Spaces.

## Why This Environment

This environment is meant for evaluating agent behavior on:

- structured workflow execution
- partial progress under dense rewards
- policy-aware response drafting
- routing and escalation decisions
- short-horizon multi-step reasoning

## Tasks

The benchmark ships with 3 deterministic tasks:

- Easy: account access / password reset
- Medium: delayed shipment with refund expectations
- Hard: enterprise outage with SLA and billing risk

Each task has a deterministic grader with scores in `[0.0, 1.0]`.

## Action Space

`SupportAction` supports these action types:

- `inspect_ticket`
- `classify_ticket`
- `set_priority`
- `route_ticket`
- `draft_reply`
- `resolve_ticket`

Relevant typed fields:

- `classification`
- `priority`
- `queue`
- `reply_text`
- `resolution_code`
- `notes`

## Observation Space

`SupportObservation` returns:

- `task_id`
- `task_difficulty`
- `customer_ticket`
- `available_queues`
- `allowed_actions`
- `history`
- `current_status`
- `partial_scores`
- `validation_errors`
- `reward_breakdown`
- `reward`
- `done`

## State Space

`SupportState` tracks:

- episode metadata
- selected task
- chosen classification / priority / queue
- current reply draft
- current resolution
- completed subgoals
- cumulative reward
- invalid action count

## Reward Design

The environment provides dense reward:

- inspecting for the first time gives a small positive reward
- correct classification, priority, and routing each add reward
- drafting a better reply increases reward according to the deterministic reply grader
- invalid or repeated no-op actions are penalized
- unsafe or forbidden reply promises are penalized
- resolving returns terminal reward aligned with the final episode score

The latest transition also exposes a typed `SupportReward` breakdown so agents and evaluators can inspect where reward came from.

## Local Setup With `uv`

```bash
uv sync --project envs/support_triage_env
```

Run the server:

```bash
uv run --project envs/support_triage_env server
```

Or:

```bash
uv run --project envs/support_triage_env uvicorn support_triage_env.server.app:app --host 0.0.0.0 --port 8000
```

## Docker

Build:

```bash
docker build -t support-triage-env:latest -f envs/support_triage_env/server/Dockerfile envs/support_triage_env
```

Run:

```bash
docker run --rm -p 8000:8000 support-triage-env:latest
```

## Validate

```bash
cd envs/support_triage_env
uv run --project . openenv validate --verbose
```

## Baseline Inference

The root-level `inference.py` uses the OpenAI Python client against an OpenAI-compatible endpoint.

Supported through `API_BASE_URL`:

- OpenAI
- Ollama with OpenAI-compatible routes
- vLLM / LM Studio / local model servers
- OpenRouter-style gateways
- Gemini only when exposed via an OpenAI-compatible endpoint

Credential lookup order:

- `OPENAI_API_KEY`
- `API_KEY`
- `HF_TOKEN`

The runner also auto-loads a repo-root `.env` file if present

The runner is intentionally tolerant of provider output and will:

- accept strict JSON replies
- extract the first JSON object from prose-wrapped replies
- record task-level errors without aborting the entire benchmark run

Example:

```bash
$env:OPENAI_API_KEY="your-key"
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
uv run python inference.py --env-url http://localhost:8000
```

## Hugging Face Spaces

From the environment directory:

```bash
openenv push --repo-id <username>/support-triage-env
```

The Space should be tagged with `openenv` and will expose:

- `/health`
- `/docs`
- `/web`
- `/ws`

Deployed Space:

- page: `https://huggingface.co/spaces/Kushal1010/support-triage-env`
- runtime: `https://kushal1010-support-triage-env.hf.space`

## Expected Baseline Output

`inference.py` prints per-task scores and an average score.

## Baseline Scores

- `MODEL_NAME=gemini-3.1-flash-lite-preview`
- `API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`
- `uv run python inference.py --env-url http://127.0.0.1:8000 --verbose`

Scores:

- `easy-password-reset`: `0.5000`
- `medium-shipping-refund`: `1.0000`
- `hard-enterprise-outage`: `0.8020`
- average: `0.7673`

Hosted verification:

- `uv run python inference.py --env-url https://kushal1010-support-triage-env.hf.space --verbose`
- matched the local baseline exactly with average `0.7673`

Output is saved in the repo root at `support_triage_baseline_results.json`.