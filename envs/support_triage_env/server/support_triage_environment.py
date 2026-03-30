"""Support triage environment implementation."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment

    from ..graders import compute_partial_scores, grade_episode
    from ..models import SupportAction, SupportObservation, SupportReward, SupportState
    from ..tasks import SupportTask, get_task_for_reset
except ImportError:  # pragma: no cover
    from openenv.core.env_server.interfaces import Environment

    from graders import compute_partial_scores, grade_episode
    from models import SupportAction, SupportObservation, SupportReward, SupportState
    from tasks import SupportTask, get_task_for_reset


class SupportTriageEnvironment(
    Environment[SupportAction, SupportObservation, SupportState]
):
    """Deterministic environment for customer support triage."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = SupportState()
        self._task: Optional[SupportTask] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional["Difficulty"] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        self._task = get_task_for_reset(task_id=task_id, difficulty=difficulty)
        self._state = SupportState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            ticket_snapshot=self._task.customer_ticket,
            partial_scores={},
            last_reward_breakdown=SupportReward(),
        )
        self._refresh_partial_scores()
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        if self._task is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self._state.step_count += 1
        errors: list[str] = []
        reward = 0.0
        reward_breakdown = SupportReward(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )

        before_episode_score = grade_episode(self._state, self._task)
        before_reply_score = self._state.partial_scores.get("reply", 0.0)

        if action.action_type == "inspect_ticket":
            if "inspected" in self._state.completed_subgoals:
                reward -= 0.05
                reward_breakdown.penalties -= 0.05
                errors.append("Ticket already inspected.")
            else:
                self._state.completed_subgoals.append("inspected")
                reward += 0.05
                reward_breakdown.inspection += 0.05

        elif action.action_type == "classify_ticket":
            self._state.current_classification = action.classification
            if action.classification == self._task.classification:
                reward += 0.20
                reward_breakdown.classification += 0.20
                self._mark_subgoal("classified")
            else:
                reward -= 0.05
                reward_breakdown.penalties -= 0.05

        elif action.action_type == "set_priority":
            self._state.current_priority = action.priority
            if action.priority == self._task.priority:
                reward += 0.15
                reward_breakdown.priority += 0.15
                self._mark_subgoal("prioritized")
            else:
                reward -= 0.05
                reward_breakdown.penalties -= 0.05
                if self._task.difficulty == "hard" and action.priority != "urgent":
                    reward -= 0.10
                    reward_breakdown.penalties -= 0.10
                    errors.append("Urgent enterprise incident was downgraded.")

        elif action.action_type == "route_ticket":
            self._state.current_queue = action.queue
            if action.queue == self._task.queue:
                reward += 0.20
                reward_breakdown.routing += 0.20
                self._mark_subgoal("routed")
            else:
                reward -= 0.05
                reward_breakdown.penalties -= 0.05

        elif action.action_type == "draft_reply":
            new_reply = (action.reply_text or "").strip()
            if not new_reply:
                reward -= 0.10
                reward_breakdown.penalties -= 0.10
                errors.append("Reply text cannot be empty.")
            elif new_reply == self._state.current_reply:
                reward -= 0.05
                reward_breakdown.penalties -= 0.05
                errors.append("Reply draft did not change.")
            else:
                self._state.current_reply = new_reply
                self._mark_subgoal("replied")
                self._refresh_partial_scores()
                reply_delta = min(
                    0.30,
                    max(
                        0.0,
                        self._state.partial_scores.get("reply", 0.0)
                        - before_reply_score,
                    ),
                )
                reward += reply_delta
                reward_breakdown.reply_delta += reply_delta
                if self._contains_forbidden_phrase():
                    reward -= 0.10
                    reward_breakdown.penalties -= 0.10
                    errors.append("Reply contains a forbidden promise or claim.")

        elif action.action_type == "resolve_ticket":
            self._state.current_resolution = action.resolution_code
            self._mark_subgoal("resolved")
            missing_core = [
                name
                for name, value in (
                    ("classification", self._state.current_classification),
                    ("priority", self._state.current_priority),
                    ("queue", self._state.current_queue),
                    ("reply", self._state.current_reply),
                )
                if not value
            ]
            allow_terminal_delta = not missing_core
            if missing_core:
                reward -= 0.15
                reward_breakdown.penalties -= 0.15
                errors.append(
                    f"Resolved before completing core fields: {', '.join(missing_core)}."
                )
        else:  # pragma: no cover
            allow_terminal_delta = True
            reward -= 0.10
            reward_breakdown.penalties -= 0.10
            errors.append(f"Unsupported action type: {action.action_type}")

        self._state.history.append(
            {
                "step": self._state.step_count,
                "action_type": action.action_type,
                "details": action.model_dump(exclude_none=True),
                "errors": list(errors),
            }
        )

        if errors:
            self._state.invalid_action_count += 1

        self._refresh_partial_scores()
        after_episode_score = self._state.partial_scores["episode"]
        done = False

        if action.action_type == "resolve_ticket":
            if allow_terminal_delta:
                terminal_delta = after_episode_score - before_episode_score
                reward += terminal_delta
                reward_breakdown.terminal_delta += terminal_delta
            done = True
        elif self._state.step_count >= self._task.max_steps:
            done = True

        reward = round(reward, 4)
        reward_breakdown.total = reward
        reward_breakdown.notes = list(errors)
        self._state.last_reward_breakdown = reward_breakdown
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)
        return self._build_observation(reward=reward, done=done, validation_errors=errors)

    @property
    def state(self) -> SupportState:
        return self._state

    def _refresh_partial_scores(self) -> None:
        if self._task is None:
            self._state.partial_scores = {}
            return
        self._state.partial_scores = compute_partial_scores(self._state, self._task)

    def _mark_subgoal(self, subgoal: str) -> None:
        if subgoal not in self._state.completed_subgoals:
            self._state.completed_subgoals.append(subgoal)

    def _contains_forbidden_phrase(self) -> bool:
        if self._task is None:
            return False
        reply_lower = self._state.current_reply.lower()
        return any(
            phrase.lower() in reply_lower for phrase in self._task.forbidden_reply_phrases
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        validation_errors: Optional[list[str]] = None,
    ) -> SupportObservation:
        if self._task is None:
            raise RuntimeError("Task is not initialized.")
        return SupportObservation(
            task_id=self._task.task_id,
            task_difficulty=self._task.difficulty,
            customer_ticket=self._task.customer_ticket,
            available_queues=[
                "billing_queue",
                "tech_queue",
                "logistics_queue",
                "trust_safety_queue",
                "enterprise_queue",
            ],
            allowed_actions=[
                "inspect_ticket",
                "classify_ticket",
                "set_priority",
                "route_ticket",
                "draft_reply",
                "resolve_ticket",
            ],
            history=list(self._state.history),
            current_status={
                "classification": self._state.current_classification,
                "priority": self._state.current_priority,
                "queue": self._state.current_queue,
                "reply_present": bool(self._state.current_reply),
                "resolution_code": self._state.current_resolution,
                "cumulative_reward": self._state.cumulative_reward,
            },
            partial_scores=dict(self._state.partial_scores),
            validation_errors=validation_errors or [],
            reward_breakdown=self._state.last_reward_breakdown,
            reward=reward,
            done=done,
            metadata={
                "task_title": self._task.title,
                "max_steps": self._task.max_steps,
                "completed_subgoals": list(self._state.completed_subgoals),
            },
        )
