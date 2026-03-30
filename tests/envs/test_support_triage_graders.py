"""Tests for support triage graders."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from support_triage_env.graders import compute_partial_scores, grade_reply
from support_triage_env.models import SupportState
from support_triage_env.tasks import get_task


def test_reply_grader_is_deterministic_under_case_and_spacing() -> None:
    task = get_task("medium-shipping-refund")
    state_a = SupportState(
        current_reply=(
            "Thank you. The delay is in transit. "
            "Our logistics team will follow up in 48 hours and review refund eligibility."
        )
    )
    state_b = SupportState(
        current_reply=(
            "  thank you. the delay is in transit. "
            "our logistics team will follow up in 48 hours and review refund eligibility. "
        )
    )

    assert grade_reply(state_a, task) == grade_reply(state_b, task)


def test_partial_scores_with_empty_state_are_bounded() -> None:
    task = get_task("easy-password-reset")
    scores = compute_partial_scores(SupportState(), task)

    assert all(0.0 <= value <= 1.0 for value in scores.values())
