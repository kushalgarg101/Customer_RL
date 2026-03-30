# Hackathon Stage 1

This repository is a standalone submission package for the `support_triage_env` OpenEnv hackathon project.

## Included

- `envs/support_triage_env/`: the environment package
- `tests/`: targeted tests for env behavior, graders, and inference helpers
- `inference.py`: root baseline runner
- `progress.md`: implementation and validation log
- `implemented_id.md`: summary of all hackathon work completed
- `.env.example`: example local model configuration
- `support_triage_baseline_results.json`: measured baseline results

## Quick Start

1. Create a local `.env` from `.env.example` and fill in your model credentials.
2. Install dependencies:

```bash
uv sync
```

3. Run the environment locally:

```bash
uv run --project envs/support_triage_env uvicorn support_triage_env.server.app:app --host 127.0.0.1 --port 8000
```

4. Run the baseline:

```bash
uv run python inference.py --env-url http://127.0.0.1:8000 --verbose
```

## Hugging Face Space

- page: `https://huggingface.co/spaces/Kushal1010/support-triage-env`
- runtime: `https://kushal1010-support-triage-env.hf.space`

## Validation

```bash
uv run pytest tests/test_support_triage_inference.py tests/envs/test_support_triage_env.py tests/envs/test_support_triage_graders.py -q
uv run --project envs/support_triage_env openenv validate envs/support_triage_env --verbose
```
