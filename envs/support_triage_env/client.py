"""Client for the support triage environment."""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SupportAction, SupportObservation, SupportState


class SupportTriageEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    """WebSocket client for support triage sessions."""

    def _step_payload(self, action: SupportAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[SupportObservation]:
        observation = SupportObservation(**payload["observation"])
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> SupportState:
        return SupportState(**payload)
