"""Tests for support triage baseline inference helpers."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent))

import inference


def test_extract_json_object_accepts_strict_json() -> None:
    payload = inference.extract_json_object('{"action_type":"inspect_ticket"}')
    assert payload == {"action_type": "inspect_ticket"}


def test_extract_json_object_accepts_wrapped_json() -> None:
    payload = inference.extract_json_object(
        "Here is the action:\n```json\n{\"action_type\": \"inspect_ticket\"}\n```"
    )
    assert payload == {"action_type": "inspect_ticket"}


def test_extract_json_object_raises_on_missing_json() -> None:
    try:
        inference.extract_json_object("no structured action here")
    except ValueError as exc:
        assert "valid JSON object" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing JSON object")


def test_normalize_action_payload_maps_common_aliases() -> None:
    payload = inference.normalize_action_payload(
        {
            "action": "classify_ticket",
            "classification": "critical_outage",
        }
    )
    assert payload["action_type"] == "classify_ticket"
    assert payload["classification"] == "enterprise_escalation"


def test_normalize_action_payload_maps_queue_and_resolution_aliases() -> None:
    payload = inference.normalize_action_payload(
        {
            "action_type": "resolve_ticket",
            "queue_name": "shipping_queue",
            "resolution": "follow_up",
        }
    )
    assert payload["queue"] == "logistics_queue"
    assert payload["resolution_code"] == "needs_followup"


def test_load_dotenv_file_sets_missing_values_only(tmp_path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "MODEL_NAME=gemini-2.5-flash\nOPENAI_API_KEY=test-key\nAPI_BASE_URL=https://example.test/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://already-set.test/v1")

    inference.load_dotenv_file(dotenv_path)

    assert os.environ["MODEL_NAME"] == "gemini-2.5-flash"
    assert os.environ["OPENAI_API_KEY"] == "test-key"
    assert os.environ["API_BASE_URL"] == "https://already-set.test/v1"


def test_emit_structured_line_flushes_and_quotes_spaces(monkeypatch) -> None:
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_print(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr("builtins.print", fake_print)

    inference.emit_structured_line("START", task="support triage eval", total_tasks=3)

    assert captured == [
        (("[START] task=\"support triage eval\" total_tasks=3",), {"flush": True})
    ]


def test_run_task_returns_error_payload_on_request_failure(monkeypatch) -> None:
    class FakeSyncEnv:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def reset(self, task_id: str):
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(task_difficulty="easy"),
            )

        def step(self, action):
            raise AssertionError("step should not be reached when request_action fails")

        def state(self):
            return SimpleNamespace(
                partial_scores={"episode": 0.75},
                step_count=0,
                cumulative_reward=0.0,
                current_classification=None,
                current_priority=None,
                current_queue=None,
                current_resolution=None,
            )

    class FakeEnvFactory:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def sync(self):
            return FakeSyncEnv()

    monkeypatch.setattr(inference, "SupportTriageEnv", FakeEnvFactory)
    monkeypatch.setattr(
        inference,
        "request_action",
        lambda client, observation: (_ for _ in ()).throw(ValueError("bad response")),
    )

    result = inference.run_task(
        client=object(),
        env_url="http://localhost:8000",
        task_id="easy-password-reset",
        max_steps=3,
    )

    assert result["score"] == 0.0
    assert result["error"] == "bad response"


def test_main_emits_structured_stdout_and_writes_results(
    monkeypatch, tmp_path, capsys
) -> None:
    output_path = tmp_path / "results.json"
    tasks = ["easy-password-reset", "medium-shipping-refund"]
    task_results = {
        "easy-password-reset": {
            "task_id": "easy-password-reset",
            "difficulty": "easy",
            "score": 0.5,
            "steps": 2,
        },
        "medium-shipping-refund": {
            "task_id": "medium-shipping-refund",
            "difficulty": "medium",
            "score": 1.0,
            "steps": 4,
            "error": "mock failure",
        },
    }

    monkeypatch.setattr(
        inference,
        "parse_args",
        lambda: SimpleNamespace(
            env_url="http://localhost:8000",
            output=str(output_path),
            max_steps=8,
            verbose=False,
        ),
    )
    monkeypatch.setenv("API_BASE_URL", "https://mock-base.test/v1")
    monkeypatch.setattr(inference, "make_client", lambda: object())
    monkeypatch.setattr(inference, "model_name", lambda: "test-model")
    monkeypatch.setattr(inference, "TASKS", tasks)
    monkeypatch.setattr(
        inference,
        "run_task",
        lambda client, env_url, task_id, max_steps, verbose: task_results[task_id],
    )

    inference.main()

    captured = capsys.readouterr()
    assert captured.err == ""

    lines = [line for line in captured.out.strip().splitlines() if line]
    assert lines == [
        "[START] task=support_triage_eval model=test-model total_tasks=2",
        "[STEP] step=1 task=easy-password-reset reward=0.5000 steps=2",
        "[STEP] step=2 task=medium-shipping-refund reward=1.0000 steps=4",
        "[END] task=support_triage_eval score=0.7500 steps=2",
    ]

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "model": "test-model",
        "api_base_url": "https://mock-base.test/v1",
        "average_score": 0.75,
        "results": [
            task_results["easy-password-reset"],
            task_results["medium-shipping-refund"],
        ],
    }
