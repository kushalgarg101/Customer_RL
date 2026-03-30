"""Deterministic graders for support triage tasks."""

from __future__ import annotations

import re

from .models import SupportState
from .tasks import SupportTask


FINAL_SCORE_WEIGHTS = {
    "classification": 0.20,
    "priority": 0.15,
    "queue": 0.20,
    "reply": 0.30,
    "resolution": 0.15,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    return any(_normalize(phrase) in normalized for phrase in phrases)


def grade_classification(state: SupportState, task: SupportTask) -> float:
    return 1.0 if state.current_classification == task.classification else 0.0


def grade_priority(state: SupportState, task: SupportTask) -> float:
    return 1.0 if state.current_priority == task.priority else 0.0


def grade_queue(state: SupportState, task: SupportTask) -> float:
    return 1.0 if state.current_queue == task.queue else 0.0


def grade_reply(state: SupportState, task: SupportTask) -> float:
    reply = state.current_reply.strip()
    if not reply:
        return 0.0

    concept_hits = 0
    for phrases in task.required_reply_concepts.values():
        if _contains_any(reply, phrases):
            concept_hits += 1

    concept_score = concept_hits / max(len(task.required_reply_concepts), 1)
    professionalism = 1.0 if any(
        term in _normalize(reply) for term in ("thank", "sorry", "understand")
    ) else 0.5
    score = concept_score * 0.8 + professionalism * 0.2

    if any(_contains_any(reply, (phrase,)) for phrase in task.forbidden_reply_phrases):
        score -= 0.3

    return max(0.0, min(1.0, round(score, 4)))


def grade_resolution(state: SupportState, task: SupportTask) -> float:
    return 1.0 if state.current_resolution == task.resolution_code else 0.0


def grade_episode(state: SupportState, task: SupportTask) -> float:
    scores = {
        "classification": grade_classification(state, task),
        "priority": grade_priority(state, task),
        "queue": grade_queue(state, task),
        "reply": grade_reply(state, task),
        "resolution": grade_resolution(state, task),
    }
    total = sum(scores[name] * weight for name, weight in FINAL_SCORE_WEIGHTS.items())
    return round(max(0.0, min(1.0, total)), 4)


def compute_partial_scores(state: SupportState, task: SupportTask) -> dict[str, float]:
    return {
        "classification": grade_classification(state, task),
        "priority": grade_priority(state, task),
        "queue": grade_queue(state, task),
        "reply": grade_reply(state, task),
        "resolution": grade_resolution(state, task),
        "episode": grade_episode(state, task),
    }
