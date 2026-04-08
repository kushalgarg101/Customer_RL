"""Unit tests for support_triage_env."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from support_triage_env.models import SupportAction
from support_triage_env.server.support_triage_environment import SupportTriageEnvironment


def _new_env() -> SupportTriageEnvironment:
    env = SupportTriageEnvironment()
    env.reset(task_id="easy-password-reset")
    return env


def test_reset_returns_initial_observation() -> None:
    env = SupportTriageEnvironment()
    obs = env.reset(task_id="easy-password-reset")

    assert obs.task_id == "easy-password-reset"
    assert obs.done is False
    assert obs.reward_breakdown.total == 0.0
    assert env.state.step_count == 0


def test_easy_task_golden_path_scores_full_episode() -> None:
    env = _new_env()

    env.step(SupportAction(action_type="inspect_ticket"))
    env.step(
        SupportAction(action_type="classify_ticket", classification="account_access")
    )
    env.step(SupportAction(action_type="set_priority", priority="medium"))
    env.step(SupportAction(action_type="route_ticket", queue="tech_queue"))
    env.step(
        SupportAction(
            action_type="draft_reply",
            reply_text=(
                "Thank you for reporting the access issue. "
                "We understand you cannot log in. "
                "We will send a new reset link so you can reset your password."
            ),
        )
    )
    result = env.step(
        SupportAction(action_type="resolve_ticket", resolution_code="needs_followup")
    )

    assert result.done is True
    assert 0.99 < result.partial_scores["episode"] < 1.0


def test_invalid_repeat_inspect_is_penalized() -> None:
    env = _new_env()

    first = env.step(SupportAction(action_type="inspect_ticket"))
    second = env.step(SupportAction(action_type="inspect_ticket"))

    assert first.reward == 0.05
    assert first.reward_breakdown.inspection == 0.05
    assert second.reward < 0
    assert second.reward_breakdown.penalties < 0
    assert second.validation_errors


def test_hard_task_wrong_priority_penalty() -> None:
    env = SupportTriageEnvironment()
    env.reset(task_id="hard-enterprise-outage")

    result = env.step(SupportAction(action_type="set_priority", priority="high"))

    assert result.reward < 0
    assert "Urgent enterprise incident was downgraded." in result.validation_errors


def test_state_tracks_updates() -> None:
    env = _new_env()
    env.step(
        SupportAction(action_type="classify_ticket", classification="account_access")
    )

    state = env.state
    assert state.step_count == 1
    assert state.current_classification == "account_access"


def test_premature_resolve_remains_penalized() -> None:
    env = _new_env()

    result = env.step(
        SupportAction(action_type="resolve_ticket", resolution_code="needs_followup")
    )

    assert result.done is True
    assert result.reward < 0
    assert any("Resolved before completing core fields" in err for err in result.validation_errors)
