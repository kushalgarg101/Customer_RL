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
        "MODEL_NAME=gemini-2.5-flash\nHF_TOKEN=test-key\nAPI_BASE_URL=https://example.test/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://already-set.test/v1")

    inference.load_dotenv_file(dotenv_path)

    assert os.environ["MODEL_NAME"] == "gemini-2.5-flash"
    assert os.environ["HF_TOKEN"] == "test-key"
    assert os.environ["API_BASE_URL"] == "https://already-set.test/v1"


def test_emit_functions_match_parser_compatible_contract(capsys) -> None:
    inference.emit_start("easy-password-reset", inference.BENCHMARK_NAME, "test-model")
    inference.emit_step(
        step=1,
        task="easy-password-reset",
        action='{"action_type":"inspect_ticket"}',
        reward=0.05,
        done=False,
        error=None,
    )
    inference.emit_end(
        task="easy-password-reset",
        success=True,
        steps=1,
        score=1.0,
        rewards=[0.05],
    )

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert lines == [
        "[START] task=easy-password-reset env=support_triage_env model=test-model",
        "[STEP] step=1 task=easy-password-reset action={\"action_type\":\"inspect_ticket\"} reward=0.05 done=false error=null",
        "[END] task=easy-password-reset success=true steps=1 score=1.00 rewards=0.05",
    ]


def test_format_action_escapes_whitespace() -> None:
    action = inference.SupportAction(
        action_type="draft_reply",
        reply_text="Hello there\ncustomer",
    )
    rendered = inference._format_action(action)
    assert " " not in rendered
    assert "\\n" in rendered
    assert "reply_text" in rendered


def test_run_task_emits_per_step_structured_lines(monkeypatch, capsys) -> None:
    class FakeSyncEnv:
        def __init__(self):
            self._steps = [
                SimpleNamespace(
                    reward=0.05,
                    done=False,
                    observation=SimpleNamespace(validation_errors=[]),
                ),
                SimpleNamespace(
                    reward=0.95,
                    done=True,
                    observation=SimpleNamespace(validation_errors=[]),
                ),
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def reset(self, task_id: str):
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(task_difficulty="easy", validation_errors=[]),
            )

        def step(self, action):
            return self._steps.pop(0)

        def state(self):
            return SimpleNamespace(
                partial_scores={"episode": 1.0},
                cumulative_reward=1.0,
                current_classification="account_access",
                current_priority="medium",
                current_queue="billing_queue",
                current_resolution="resolved",
            )

    class FakeEnvFactory:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def sync(self):
            return FakeSyncEnv()

    actions = iter(
        [
            (
                inference.SupportAction(action_type="inspect_ticket"),
                '{"action_type":"inspect_ticket"}',
                {"action_type": "inspect_ticket"},
            ),
            (
                inference.SupportAction(
                    action_type="resolve_ticket", resolution_code="resolved"
                ),
                '{"action_type":"resolve_ticket","resolution_code":"resolved"}',
                {
                    "action_type": "resolve_ticket",
                    "resolution_code": "resolved",
                },
            ),
        ]
    )

    monkeypatch.setattr(inference, "SupportTriageEnv", FakeEnvFactory)
    monkeypatch.setattr(inference, "MODEL_NAME", "test-model")
    monkeypatch.setattr(inference, "request_action", lambda client, observation: next(actions))

    result = inference.run_task(
        client=object(),
        env_url="http://localhost:8000",
        task_id="easy-password-reset",
        max_steps=3,
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert [line for line in captured.out.strip().splitlines() if line] == [
        "[START] task=easy-password-reset env=support_triage_env model=test-model",
        "[STEP] step=1 task=easy-password-reset action={\"action_type\":\"inspect_ticket\"} reward=0.05 done=false error=null",
        "[STEP] step=2 task=easy-password-reset action={\"action_type\":\"resolve_ticket\",\"resolution_code\":\"resolved\"} reward=0.95 done=true error=null",
        "[END] task=easy-password-reset success=true steps=2 score=1.00 rewards=0.05,0.95",
    ]
    assert result["score"] == 1.0
    assert result["success"] is True


def test_run_task_emits_fallback_step_on_failure_before_env_step(monkeypatch, capsys) -> None:
    class FakeSyncEnv:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def reset(self, task_id: str):
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(task_difficulty="easy", validation_errors=[]),
            )

        def state(self):
            return SimpleNamespace(
                partial_scores={"episode": 0.0},
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
    monkeypatch.setattr(inference, "MODEL_NAME", "test-model")
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

    captured = capsys.readouterr()
    assert [line for line in captured.out.strip().splitlines() if line] == [
        "[START] task=easy-password-reset env=support_triage_env model=test-model",
        "[STEP] step=1 task=easy-password-reset action=null reward=0.00 done=true error=bad\\u0020response",
        "[END] task=easy-password-reset success=false steps=0 score=0.00 rewards=",
    ]
    assert result["score"] == 0.0
    assert result["success"] is False
    assert result["error"] == "bad response"


def test_run_task_preserves_partial_score_after_late_failure(monkeypatch, capsys) -> None:
    class FakeSyncEnv:
        def __init__(self):
            self._steps = [
                SimpleNamespace(
                    reward=0.25,
                    done=False,
                    observation=SimpleNamespace(validation_errors=[]),
                )
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def reset(self, task_id: str):
            return SimpleNamespace(
                done=False,
                observation=SimpleNamespace(task_difficulty="easy", validation_errors=[]),
            )

        def step(self, action):
            return self._steps.pop(0)

        def state(self):
            return SimpleNamespace(
                partial_scores={"episode": 0.25},
                cumulative_reward=0.25,
                current_classification="account_access",
                current_priority=None,
                current_queue=None,
                current_resolution=None,
            )

    class FakeEnvFactory:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def sync(self):
            return FakeSyncEnv()

    actions = iter(
        [
            (
                inference.SupportAction(action_type="inspect_ticket"),
                '{"action_type":"inspect_ticket"}',
                {"action_type": "inspect_ticket"},
            )
        ]
    )

    def fake_request_action(client, observation):
        try:
            return next(actions)
        except StopIteration as exc:
            raise ValueError("late failure") from exc

    monkeypatch.setattr(inference, "SupportTriageEnv", FakeEnvFactory)
    monkeypatch.setattr(inference, "MODEL_NAME", "test-model")
    monkeypatch.setattr(inference, "request_action", fake_request_action)

    result = inference.run_task(
        client=object(),
        env_url="http://localhost:8000",
        task_id="easy-password-reset",
        max_steps=3,
    )

    captured = capsys.readouterr()
    assert [line for line in captured.out.strip().splitlines() if line] == [
        "[START] task=easy-password-reset env=support_triage_env model=test-model",
        "[STEP] step=1 task=easy-password-reset action={\"action_type\":\"inspect_ticket\"} reward=0.25 done=false error=null",
        "[END] task=easy-password-reset success=false steps=1 score=0.25 rewards=0.25",
    ]
    assert result["score"] == 0.25
    assert result["success"] is False
    assert result["error"] == "late failure"


def test_run_task_emits_start_step_and_end_when_client_init_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(inference, "MODEL_NAME", "test-model")
    monkeypatch.setattr(
        inference,
        "make_client",
        lambda: (_ for _ in ()).throw(RuntimeError("missing token")),
    )

    result = inference.run_task(
        client=None,
        env_url="http://localhost:8000",
        task_id="easy-password-reset",
        max_steps=3,
    )

    captured = capsys.readouterr()
    assert [line for line in captured.out.strip().splitlines() if line] == [
        "[START] task=easy-password-reset env=support_triage_env model=test-model",
        "[STEP] step=1 task=easy-password-reset action=null reward=0.00 done=true error=missing\\u0020token",
        "[END] task=easy-password-reset success=false steps=0 score=0.00 rewards=",
    ]
    assert result["score"] == 0.0
    assert result["success"] is False
    assert result["error"] == "missing token"


def test_main_writes_results_without_extra_stdout(monkeypatch, tmp_path, capsys) -> None:
    output_path = tmp_path / "results.json"
    task_results = {
        "easy-password-reset": {
            "task_id": "easy-password-reset",
            "difficulty": "easy",
            "score": 0.5,
            "steps": 2,
            "rewards": [0.05, 0.45],
            "cumulative_reward": 0.5,
            "classification": None,
            "priority": None,
            "queue": None,
            "resolution": None,
            "success": False,
        }
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
    monkeypatch.setattr(inference, "HF_TOKEN", "test-token")
    monkeypatch.setattr(inference, "API_BASE_URL", "https://mock-base.test/v1")
    monkeypatch.setattr(inference, "MODEL_NAME", "test-model")
    monkeypatch.setattr(inference, "TASKS", ["easy-password-reset"])
    monkeypatch.setattr(
        inference,
        "run_task",
        lambda client, env_url, task_id, max_steps, verbose: task_results[task_id],
    )

    inference.main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "model": "test-model",
        "api_base_url": "https://mock-base.test/v1",
        "average_score": 0.5,
        "results": [task_results["easy-password-reset"]],
    }
