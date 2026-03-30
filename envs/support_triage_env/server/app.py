"""FastAPI app for the support triage environment."""

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import SupportAction, SupportObservation
    from .support_triage_environment import SupportTriageEnvironment
except ImportError:  # pragma: no cover
    from openenv.core.env_server.http_server import create_app

    from models import SupportAction, SupportObservation
    from server.support_triage_environment import SupportTriageEnvironment


app = create_app(
    SupportTriageEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_triage_env",
    max_concurrent_envs=8,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
