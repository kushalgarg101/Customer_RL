"""Typed models for the support triage environment."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:  # pragma: no cover
    from openenv.core.env_server.interfaces import Action, Observation, State

ActionType = Literal[
    "inspect_ticket",
    "classify_ticket",
    "set_priority",
    "route_ticket",
    "draft_reply",
    "resolve_ticket",
]
Classification = Literal[
    "billing",
    "technical",
    "shipping",
    "account_access",
    "enterprise_escalation",
]
Priority = Literal["low", "medium", "high", "urgent"]
QueueName = Literal[
    "billing_queue",
    "tech_queue",
    "logistics_queue",
    "trust_safety_queue",
    "enterprise_queue",
]
ResolutionCode = Literal["resolved", "needs_followup", "escalated"]
Difficulty = Literal["easy", "medium", "hard"]


class SupportAction(Action):
    """A structured action in the support workflow."""

    action_type: ActionType
    classification: Optional[Classification] = None
    priority: Optional[Priority] = None
    queue: Optional[QueueName] = None
    reply_text: Optional[str] = None
    resolution_code: Optional[ResolutionCode] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SupportAction":
        required_by_action = {
            "classify_ticket": ("classification",),
            "set_priority": ("priority",),
            "route_ticket": ("queue",),
            "draft_reply": ("reply_text",),
            "resolve_ticket": ("resolution_code",),
        }
        for field_name in required_by_action.get(self.action_type, ()):
            if getattr(self, field_name) in (None, ""):
                raise ValueError(f"{field_name} is required for {self.action_type}")
        return self


class SupportObservation(Observation):
    """Agent-facing observation for the support workflow."""

    task_id: str
    task_difficulty: Difficulty
    customer_ticket: str
    available_queues: list[QueueName]
    allowed_actions: list[ActionType]
    history: list[dict[str, Any]] = Field(default_factory=list)
    current_status: dict[str, Any] = Field(default_factory=dict)
    partial_scores: dict[str, float] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    reward_breakdown: "SupportReward" = Field(default_factory=lambda: SupportReward())


class SupportReward(State):
    """Typed reward breakdown for the most recent transition."""

    episode_id: Optional[str] = None
    step_count: int = 0
    total: float = 0.0
    inspection: float = 0.0
    classification: float = 0.0
    priority: float = 0.0
    routing: float = 0.0
    reply_delta: float = 0.0
    terminal_delta: float = 0.0
    penalties: float = 0.0
    notes: list[str] = Field(default_factory=list)


class SupportState(State):
    """Full server-side state for the support workflow."""

    task_id: Optional[str] = None
    ticket_snapshot: str = ""
    current_classification: Optional[Classification] = None
    current_priority: Optional[Priority] = None
    current_queue: Optional[QueueName] = None
    current_reply: str = ""
    current_resolution: Optional[ResolutionCode] = None
    completed_subgoals: list[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    invalid_action_count: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)
    partial_scores: dict[str, float] = Field(default_factory=dict)
    last_reward_breakdown: SupportReward = Field(default_factory=SupportReward)
