"""Baseline inference runner for support_triage_env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent


def _bootstrap_repo_imports() -> None:
    """Allow running from repo root without requiring editable installs."""
    for candidate in (REPO_ROOT / "envs", REPO_ROOT / "src"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_repo_imports()

from support_triage_env import SupportAction, SupportTriageEnv
from support_triage_env.prompts import SYSTEM_PROMPT
from support_triage_env.tasks import TASKS


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OUTPUT_PATH = Path("support_triage_baseline_results.json")
ACTION_ALIASES = {
    "action": "action_type",
    "type": "action_type",
    "reply": "reply_text",
    "message": "reply_text",
    "resolution": "resolution_code",
    "resolution_status": "resolution_code",
    "queue_name": "queue",
}
CLASSIFICATION_ALIASES = {
    "shipping_delay": "shipping",
    "shipping_issue": "shipping",
    "delivery_issue": "shipping",
    "refund_request": "billing",
    "billing_issue": "billing",
    "tech_support": "technical",
    "technical_issue": "technical",
    "login_issue": "account_access",
    "password_reset": "account_access",
    "account_issue": "account_access",
    "critical_outage": "enterprise_escalation",
    "enterprise_outage": "enterprise_escalation",
    "outage": "enterprise_escalation",
}
PRIORITY_ALIASES = {
    "critical": "urgent",
    "p1": "urgent",
    "p2": "high",
    "normal": "medium",
}
QUEUE_ALIASES = {
    "billing": "billing_queue",
    "billing_support": "billing_queue",
    "technical": "tech_queue",
    "technical_support": "tech_queue",
    "support_queue": "tech_queue",
    "shipping_queue": "logistics_queue",
    "logistics": "logistics_queue",
    "enterprise_support": "enterprise_queue",
    "escalations": "enterprise_queue",
}
RESOLUTION_ALIASES = {
    "follow_up": "needs_followup",
    "needs_follow_up": "needs_followup",
    "escalate": "escalated",
    "escalate_case": "escalated",
    "closed": "resolved",
    "close": "resolved",
}


def load_dotenv_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries from a repo-local .env file."""
    dotenv_path = path or (REPO_ROOT / ".env")
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


load_dotenv_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline inference for support_triage_env."
    )
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def make_client() -> OpenAI:
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("HF_TOKEN")
    )
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY, API_KEY, or HF_TOKEN.")
    base_url = os.environ.get("API_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def model_name() -> str:
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


def build_messages(observation: Any) -> list[dict[str, str]]:
    payload = {
        "task_id": observation.task_id,
        "ticket": observation.customer_ticket,
        "allowed_actions": observation.allowed_actions,
        "available_queues": observation.available_queues,
        "current_status": observation.current_status,
        "partial_scores": observation.partial_scores,
        "history": observation.history,
        "validation_errors": observation.validation_errors,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Choose the next action for the support workflow.\n"
                "Return strict JSON only.\n\n"
                f"{json.dumps(payload, indent=2)}"
            ),
        },
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response."""
    stripped = text.strip()
    decoder = json.JSONDecoder()

    for candidate in (stripped, stripped.replace("```json", "").replace("```", "").strip()):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError("Model response did not contain a valid JSON object.")


def normalize_action_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Map common model near-misses into the exact SupportAction schema."""
    normalized = dict(data)

    for old_key, new_key in ACTION_ALIASES.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(old_key)

    if "action_type" in normalized and isinstance(normalized["action_type"], str):
        normalized["action_type"] = normalized["action_type"].strip()

    if "classification" in normalized and isinstance(normalized["classification"], str):
        normalized["classification"] = CLASSIFICATION_ALIASES.get(
            normalized["classification"].strip(), normalized["classification"].strip()
        )

    if "priority" in normalized and isinstance(normalized["priority"], str):
        normalized["priority"] = PRIORITY_ALIASES.get(
            normalized["priority"].strip().lower(),
            normalized["priority"].strip().lower(),
        )

    if "queue" in normalized and isinstance(normalized["queue"], str):
        normalized["queue"] = QUEUE_ALIASES.get(
            normalized["queue"].strip().lower(),
            normalized["queue"].strip().lower(),
        )

    if "resolution_code" in normalized and isinstance(
        normalized["resolution_code"], str
    ):
        normalized["resolution_code"] = RESOLUTION_ALIASES.get(
            normalized["resolution_code"].strip().lower(),
            normalized["resolution_code"].strip().lower(),
        )

    return normalized


def request_action(
    client: OpenAI, observation: Any
) -> tuple[SupportAction, str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=model_name(),
        temperature=0,
        messages=build_messages(observation),
    )
    content = response.choices[0].message.content or "{}"
    data = extract_json_object(content)
    normalized = normalize_action_payload(data)
    return SupportAction(**normalized), content, normalized


def run_task(
    client: OpenAI, env_url: str, task_id: str, max_steps: int, verbose: bool = False
) -> dict[str, Any]:
    with SupportTriageEnv(base_url=env_url).sync() as env:
        result = env.reset(task_id=task_id)
        final_result = result
        error_message: str | None = None
        last_raw_response: str | None = None
        last_normalized_action: dict[str, Any] | None = None
        for _ in range(max_steps):
            if final_result.done:
                break
            try:
                action, raw_response, normalized_action = request_action(
                    client, final_result.observation
                )
                last_raw_response = raw_response
                last_normalized_action = normalized_action
                if verbose:
                    print(f"[{task_id}] raw model response: {raw_response}")
                    print(
                        f"[{task_id}] normalized action: "
                        f"{json.dumps(normalized_action, ensure_ascii=False)}"
                    )
                final_result = env.step(action)
            except Exception as exc:
                error_message = str(exc)
                if verbose:
                    print(f"[{task_id}] error: {error_message}")
                    if last_raw_response:
                        print(f"[{task_id}] last raw response: {last_raw_response}")
                    if last_normalized_action:
                        print(
                            f"[{task_id}] last normalized action: "
                            f"{json.dumps(last_normalized_action, ensure_ascii=False)}"
                        )
                break
        state = env.state()
        payload = {
            "task_id": task_id,
            "difficulty": result.observation.task_difficulty,
            "score": 0.0 if error_message else state.partial_scores.get("episode", 0.0),
            "steps": state.step_count,
            "cumulative_reward": state.cumulative_reward,
            "classification": state.current_classification,
            "priority": state.current_priority,
            "queue": state.current_queue,
            "resolution": state.current_resolution,
        }
        if error_message:
            payload["error"] = error_message
        if last_raw_response:
            payload["last_raw_response"] = last_raw_response
        if last_normalized_action:
            payload["last_normalized_action"] = last_normalized_action
        return payload


def main() -> None:
    args = parse_args()
    client = make_client()

    results = [
        run_task(client, args.env_url, task_id, args.max_steps, verbose=args.verbose)
        for task_id in TASKS
    ]
    average = round(sum(item["score"] for item in results) / len(results), 4)
    payload = {
        "model": model_name(),
        "api_base_url": os.environ.get("API_BASE_URL", DEFAULT_BASE_URL),
        "average_score": average,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Model: {payload['model']}")
    for item in results:
        print(
            f"- {item['task_id']} ({item['difficulty']}): "
            f"score={item['score']:.4f}, steps={item['steps']}"
        )
        if "error" in item:
            print(f"  error: {item['error']}")
    print(f"Average score: {average:.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
