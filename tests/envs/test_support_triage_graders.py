"""Tests for support triage graders."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from support_triage_env.graders import (
    STRICT_SCORE_EPSILON,
    compute_partial_scores,
    grade_classification,
    grade_episode,
    grade_priority,
    grade_queue,
    grade_reply,
    grade_resolution,
)
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

    assert all(0.0 < value < 1.0 for value in scores.values())
    assert all(value == STRICT_SCORE_EPSILON for value in scores.values())


def test_perfect_state_scores_are_near_one_but_not_one() -> None:
    task = get_task("easy-password-reset")
    state = SupportState(
        current_classification=task.classification,
        current_priority=task.priority,
        current_queue=task.queue,
        current_reply=(
            "Thank you for reporting the access issue. "
            "We understand you cannot log in. "
            "We will send a new reset link so you can reset your password."
        ),
        current_resolution=task.resolution_code,
    )

    scores = compute_partial_scores(state, task)

    assert all(0.0 < value < 1.0 for value in scores.values())
    assert scores["classification"] == round(1.0 - STRICT_SCORE_EPSILON, 4)
    assert scores["priority"] == round(1.0 - STRICT_SCORE_EPSILON, 4)
    assert scores["queue"] == round(1.0 - STRICT_SCORE_EPSILON, 4)
    assert scores["resolution"] == round(1.0 - STRICT_SCORE_EPSILON, 4)
    assert 0.99 < scores["episode"] < 1.0


def test_binary_graders_no_longer_return_exact_endpoints() -> None:
    task = get_task("hard-enterprise-outage")
    empty_state = SupportState()
    perfect_state = SupportState(
        current_classification=task.classification,
        current_priority=task.priority,
        current_queue=task.queue,
        current_resolution=task.resolution_code,
    )

    for grader in (
        grade_classification,
        grade_priority,
        grade_queue,
        grade_resolution,
    ):
        assert grader(empty_state, task) == STRICT_SCORE_EPSILON
        assert grader(perfect_state, task) == round(1.0 - STRICT_SCORE_EPSILON, 4)


def test_episode_score_with_mixed_progress_remains_interior() -> None:
    task = get_task("medium-shipping-refund")
    state = SupportState(
        current_classification=task.classification,
        current_priority="medium",
        current_queue=task.queue,
        current_reply=(
            "Thank you. The delay is in transit. "
            "Our logistics team will follow up in 48 hours and review refund eligibility."
        ),
        current_resolution="needs_followup",
    )

    score = grade_episode(state, task)

    assert 0.0 < score < 1.0
    assert score not in (STRICT_SCORE_EPSILON, round(1.0 - STRICT_SCORE_EPSILON, 4))
