"""Deterministic support triage task fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Classification, Difficulty, Priority, QueueName, ResolutionCode


@dataclass(frozen=True)
class SupportTask:
    task_id: str
    difficulty: Difficulty
    title: str
    customer_ticket: str
    classification: Classification
    priority: Priority
    queue: QueueName
    resolution_code: ResolutionCode
    required_reply_concepts: dict[str, tuple[str, ...]]
    forbidden_reply_phrases: tuple[str, ...]
    max_steps: int


TASKS: dict[str, SupportTask] = {
    "easy-password-reset": SupportTask(
        task_id="easy-password-reset",
        difficulty="easy",
        title="Password Reset Access Issue",
        customer_ticket=(
            "Customer: I cannot log into my account after changing phones. "
            "The password reset link expired twice. Please help me regain access."
        ),
        classification="account_access",
        priority="medium",
        queue="tech_queue",
        resolution_code="needs_followup",
        required_reply_concepts={
            "acknowledge_access_issue": ("access", "log in"),
            "reset_guidance": ("reset", "password"),
            "next_step": ("new link", "reset link"),
        },
        forbidden_reply_phrases=("refund", "shipping", "guarantee"),
        max_steps=6,
    ),
    "medium-shipping-refund": SupportTask(
        task_id="medium-shipping-refund",
        difficulty="medium",
        title="Delayed Shipment With Refund Request",
        customer_ticket=(
            "Customer: My order was supposed to arrive last week and still shows in transit. "
            "This was a birthday gift. If it will not arrive in two days I want a refund. "
            "Please tell me what happens next."
        ),
        classification="shipping",
        priority="high",
        queue="logistics_queue",
        resolution_code="needs_followup",
        required_reply_concepts={
            "acknowledge_delay": ("delay", "in transit"),
            "logistics_next_step": ("shipping team", "logistics"),
            "refund_policy": ("refund", "eligible"),
            "follow_up_timeline": ("follow up", "48 hours"),
        },
        forbidden_reply_phrases=("password", "sla", "guaranteed refund today"),
        max_steps=7,
    ),
    "hard-enterprise-outage": SupportTask(
        task_id="hard-enterprise-outage",
        difficulty="hard",
        title="Enterprise Outage With Billing Risk",
        customer_ticket=(
            "Enterprise customer: Since 07:10 UTC our finance team cannot export invoices "
            "for month-end close. We have 120 blocked users across regions and may miss SLA "
            "commitments. Our account team mentioned billing credits if downtime exceeds the SLA. "
            "We need immediate escalation and a written update."
        ),
        classification="enterprise_escalation",
        priority="urgent",
        queue="enterprise_queue",
        resolution_code="escalated",
        required_reply_concepts={
            "acknowledge_urgency": ("urgent", "immediate"),
            "escalation_path": ("escalated", "enterprise"),
            "status_update": ("update", "incident"),
            "billing_handling": ("billing", "credit"),
            "follow_up_timeline": ("30 minutes", "next update"),
        },
        forbidden_reply_phrases=("issue resolved", "low priority", "guaranteed credit"),
        max_steps=8,
    ),
}


TASKS_BY_DIFFICULTY: dict[Difficulty, SupportTask] = {
    task.difficulty: task for task in TASKS.values()
}


def get_task(task_id: str) -> SupportTask:
    return TASKS[task_id]


def get_task_for_reset(
    task_id: str | None = None, difficulty: Difficulty | None = None
) -> SupportTask:
    if task_id is not None:
        return get_task(task_id)
    if difficulty is not None:
        return TASKS_BY_DIFFICULTY[difficulty]
    return TASKS_BY_DIFFICULTY["easy"]
