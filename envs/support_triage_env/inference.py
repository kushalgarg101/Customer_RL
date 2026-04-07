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
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_OUTPUT_PATH = Path("support_triage_baseline_results.json")
BENCHMARK_NAME = "support_triage_env"
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

HF_TOKEN = (
    os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


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
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN, OPENAI_API_KEY, or API_KEY.")
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def debug_log(message: str, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _escape_log_text(text: str) -> str:
    escaped: list[str] = []
    for char in text:
        if char == " ":
            escaped.append("\\u0020")
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif char.isspace():
            escaped.append(f"\\u{ord(char):04x}")
        else:
            escaped.append(char)
    return "".join(escaped)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(message: str | None) -> str:
    if not message:
        return "null"
    return _escape_log_text(message)


def _format_action(action: SupportAction) -> str:
    raw = json.dumps(action.model_dump(exclude_none=True, exclude={"metadata"}, exclude_defaults=True), separators=(",", ":"), ensure_ascii=True)
    return _escape_log_text(raw)


def _clamp_score(score: float) -> float:
    return min(max(score, 0.0), 1.0)


def _extract_step_error(observation: Any) -> str | None:
    validation_errors = getattr(observation, "validation_errors", None) or []
    if not validation_errors:
        return None
    return "; ".join(str(item) for item in validation_errors)


def emit_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def emit_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_format_bool(done)} error={_format_error(error)}",
        flush=True,
    )


def emit_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={_format_bool(success)} steps={steps} "
        f"score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


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

    for candidate in (
        stripped,
        stripped.replace("```json", "").replace("```", "").strip(),
    ):
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

    if "resolution_code" in normalized and isinstance(normalized["resolution_code"], str):
        normalized["resolution_code"] = RESOLUTION_ALIASES.get(
            normalized["resolution_code"].strip().lower(),
            normalized["resolution_code"].strip().lower(),
        )

    return normalized


def request_action(
    client: OpenAI, observation: Any
) -> tuple[SupportAction, str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=build_messages(observation),
    )
    content = response.choices[0].message.content or "{}"
    data = extract_json_object(content)
    normalized = normalize_action_payload(data)
    return SupportAction(**normalized), content, normalized


def run_task(
    client: OpenAI | None, env_url: str, task_id: str, max_steps: int, verbose: bool = False
) -> dict[str, Any]:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_message: str | None = None
    difficulty: str | None = None
    last_raw_response: str | None = None
    last_normalized_action: dict[str, Any] | None = None
    state: Any | None = None
    env: Any | None = None

    emit_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        if client is None:
            client = make_client()
        with SupportTriageEnv(base_url=env_url).sync() as env:
            result = env.reset(task_id=task_id)
            difficulty = result.observation.task_difficulty
            final_result = result

            for step in range(1, max_steps + 1):
                if final_result.done:
                    break

                action, raw_response, normalized_action = request_action(
                    client, final_result.observation
                )
                last_raw_response = raw_response
                last_normalized_action = normalized_action
                debug_log(f"[{task_id}] raw model response: {raw_response}", verbose)
                debug_log(
                    "[{}] normalized action: {}".format(
                        task_id,
                        json.dumps(normalized_action, ensure_ascii=False),
                    ),
                    verbose,
                )

                final_result = env.step(action)
                reward = float(final_result.reward or 0.0)
                rewards.append(reward)
                steps_taken = step
                step_error = _extract_step_error(final_result.observation)
                if step_error:
                    error_message = step_error
                emit_step(
                    step=step,
                    action=_format_action(action),
                    reward=reward,
                    done=bool(final_result.done),
                    error=step_error,
                )

                if final_result.done:
                    break

            state = env.state()
            score = _clamp_score(float(state.partial_scores.get("episode", 0.0)))
            success = score >= 1.0 and error_message is None
    except Exception as exc:
        if error_message is None:
            error_message = str(exc)
        if state is None and env is not None:
            try:
                state = env.state()
                score = _clamp_score(float(state.partial_scores.get("episode", 0.0)))
            except Exception:
                state = None
        debug_log(f"[{task_id}] error: {error_message}", verbose)
        if last_raw_response:
            debug_log(f"[{task_id}] last raw response: {last_raw_response}", verbose)
        if last_normalized_action:
            debug_log(
                "[{}] last normalized action: {}".format(
                    task_id,
                    json.dumps(last_normalized_action, ensure_ascii=False),
                ),
                verbose,
            )
    finally:
        emit_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    payload = {
        "task_id": task_id,
        "difficulty": difficulty,
        "score": score,
        "steps": steps_taken,
        "rewards": rewards,
        "cumulative_reward": getattr(state, "cumulative_reward", 0.0),
        "classification": getattr(state, "current_classification", None),
        "priority": getattr(state, "current_priority", None),
        "queue": getattr(state, "current_queue", None),
        "resolution": getattr(state, "current_resolution", None),
        "success": success,
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

    results = [
        run_task(None, args.env_url, task_id, args.max_steps, verbose=args.verbose)
        for task_id in TASKS
    ]
    average = round(sum(item["score"] for item in results) / len(results), 4)
    payload = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "average_score": average,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()





