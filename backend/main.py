from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from exceptions.registry_execeptions import RegistryException
from experiments.run import Experiment, ModelCard
from registry.agent_registry import AgentRegistry
from registry.agent_instance_registry import AgentInstanceRegistry
from registry.experiment_registry import ExperimentRegistry
from registry.env_registry import EnvironmentRegistry
from agents.policy_iteration import (
    PolicyIterationAgent,
    PolicyIterationAgentParameters,
    PolicyIterationTrainingParameters,
)

from typing import Any
from utils.models import ActionResult

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ------------------------- Request Schemas -------------------------

class ExperimentRequest(BaseModel):
    agent_type: str
    experiment_name: str
    env_type: str
    agent_params: dict[str, Any] = Field(default_factory=dict)
    # kept for frontend compatibility; not stored on the Experiment anymore
    env_config: dict[str, Any] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    experiment_id: str
    train_params: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    experiment_id: str
    state: int


# ------------------------- Response Schemas -------------------------

class HealthResponse(BaseModel):
    status: str


class RegisterExperimentResponse(BaseModel):
    success: bool
    experiment_id: str
    env_id: str


class ExperimentSummary(BaseModel):
    experiment_id: str
    experiment_name: str
    env_type: str
    agent_type: str
    trained: bool
    env_id: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentSummary]
    success: bool


class ModelCardResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    env_type: str
    agent_type: str
    agent_params: dict[str, Any]
    training_params: dict[str, Any]
    trained: bool
    value: list[float] | None = None
    env_id: str


class ExperimentFilterResponse(BaseModel):
    experiments: list[ModelCardResponse]
    success: bool


class ExperimentDetailResponse(BaseModel):
    success: bool
    experiment_id: str
    experiment_name: str
    env_type: str
    agent_type: str
    trained: bool
    value: list[float] | None = None
    agent_params: dict[str, Any]
    training_params: dict[str, Any] | None = None
    env_id: str


class EnvironmentSummary(BaseModel):
    env_id: str
    config: dict[str, Any]


class EnvironmentListResponse(BaseModel):
    environments: list[EnvironmentSummary]
    success: bool


class TrainResponse(BaseModel):
    value: list[float]
    success: bool


class PredictResponse(BaseModel):
    action: int


# ------------------------- Registries -------------------------

# Type catalog: how to build/train each agent kind (keyed by agent_type).
agent_registry = AgentRegistry()
agent_registry.register(
    "PolicyIteration",
    PolicyIterationAgent,
    PolicyIterationTrainingParameters,
    PolicyIterationAgentParameters,
)

# Runtime store: initialized agent objects (keyed by agent_id).
agent_instances = AgentInstanceRegistry(agent_registry)

# Experiments keyed by experiment_id.
experiments = ExperimentRegistry()

# Environments keyed by env_id (deduplicated by structural fingerprint).
environments = EnvironmentRegistry()


# ------------------------- JSON helpers -------------------------

def _model_card_response(card: ModelCard) -> ModelCardResponse:
    return ModelCardResponse(
        experiment_id=card.experiment_id,
        experiment_name=card.experiment_name,
        env_type=card.env_type,
        agent_type=card.agent_type,
        agent_params=card.agent_params,
        training_params=card.training_params,
        trained=card.trained,
        value=card.value_function.tolist() if card.value_function is not None else None,
        env_id=card.env_id,
    )


def _ensure_env_id(experiment: Experiment) -> str:
    """Derive + persist an env_id for experiments lacking one (backward compat)."""
    if not experiment.env_id:
        env_config = dict(
            (experiment.agent_params or {}).get("env_config") or {}
        )
        experiment.env_id = environments.get_or_create(env_config)
    return experiment.env_id


def normalize_transitions(P: dict[str, Any]) -> dict[int, dict[int, list[ActionResult]]]:
    """Convert a JSON-shaped P table (string keys, dict outcomes) into the
    int-keyed ActionResult form consumed by the agents."""

    def _as_action_result(outcome: Any) -> ActionResult:
        if isinstance(outcome, ActionResult):
            return outcome
        return ActionResult(
            prob=float(outcome.get("prob")),
            next_state=int(outcome.get("next_state")),
            reward=float(outcome.get("reward")),
            done=bool(outcome.get("done")),
        )

    return {
        int(s): {
            int(a): [_as_action_result(o) for o in outcomes]
            for a, outcomes in actions.items()
        }
        for s, actions in P.items()
    }


def _create_experiment(req: ExperimentRequest) -> Experiment:
    """Get-or-create the agent and build a new Experiment from the request."""
    experiment_id = str(uuid4())

    # Agent init params, with experiment-level defaults applied.
    agent_params = dict(req.agent_params or {})
    agent_params.setdefault("name", req.experiment_name)
    agent_params.setdefault("env_type", req.env_type)

    # Environment config may arrive top-level or nested inside agent_params
    # (frontend compatibility). Store it under one canonical key and give it
    # a stable env_id (reused for structurally equivalent environments).
    env_config = dict(agent_params.get("env_config") or req.env_config or {})
    agent_params["env_config"] = env_config
    env_id = environments.get_or_create(env_config)

    # Resolve the type entry once, then reuse it for both the agent and train params.
    entry = agent_registry.get_agent(req.agent_type)

    agent = agent_instances.get_or_create(
        agent_id=experiment_id,      # 1 agent per experiment by default
        agent_type=req.agent_type,
        agent_params=agent_params,
    )

    return Experiment(
        experiment_id=experiment_id,
        experiment_name=req.experiment_name,
        env_type=req.env_type,
        agent_type=req.agent_type,
        agent=agent,
        agent_params=agent_params,
        env_id=env_id,
        training_parameter_class=entry.training_parameter_class,
    )


# ------------------------- Endpoints -------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return server liveness only."""
    return HealthResponse(status="alive")


@app.post("/register_experiment", response_model=RegisterExperimentResponse)
def register_experiment(req: ExperimentRequest) -> RegisterExperimentResponse:
    """Get-or-create the agent and register a new experiment."""
    try:
        experiment = _create_experiment(req)
        experiments.register_experiment(experiment)
        return RegisterExperimentResponse(
            success=True,
            experiment_id=experiment.experiment_id,
            env_id=experiment.env_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")


def _get_experiment(experiment_id: str) -> Experiment:
    try:
        return experiments.retrieve(experiment_id)
    except RegistryException as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/experiments", response_model=ExperimentListResponse)
def list_experiments() -> ExperimentListResponse:
    """List every registered experiment (session) from the registry."""
    return ExperimentListResponse(
        experiments=[
            ExperimentSummary(
                experiment_id=card.experiment_id,
                experiment_name=card.experiment_name,
                env_type=card.env_type,
                agent_type=card.agent_type,
                trained=card.trained,
                env_id=_ensure_env_id(experiments.retrieve(card.experiment_id)),
            )
            for card in experiments.get_model_cards()
        ],
        success=True,
    )


# NOTE: must be declared BEFORE /experiments/{experiment_id} so "filter"
# isn't captured as an experiment_id path segment.
@app.get("/experiments/filter", response_model=ExperimentFilterResponse)
def filter_experiments(
    agent_type: str | None = None,
    env_type: str | None = None,
    trained: bool | None = None,
    env_id: str | None = None,
) -> ExperimentFilterResponse:
    """Filter experiments by env_id / agent_type / env_type (+ optional trained flag)."""
    cards = experiments.filter(
        env_type=env_type,
        agent_type=agent_type,
        trained=trained,
        env_id=env_id,
    )
    return ExperimentFilterResponse(
        experiments=[_model_card_response(c) for c in cards],
        success=True,
    )


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    """Return a single registered experiment's details + state value function."""
    experiment = _get_experiment(experiment_id)
    env_id = _ensure_env_id(experiment)
    return ExperimentDetailResponse(
        success=True,
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.experiment_name,
        env_type=experiment.env_type,
        agent_type=experiment.agent_type,
        trained=experiment.trained,
        value=experiment.value_function.tolist()
        if experiment.value_function is not None else None,
        agent_params=experiment.agent_params,
        training_params=experiment.training_params,
        env_id=env_id,
    )


@app.get("/envs", response_model=EnvironmentListResponse)
def list_environments() -> EnvironmentListResponse:
    """List every stored environment with its full config."""
    return EnvironmentListResponse(
        environments=[
            EnvironmentSummary(env_id=env_id, config=config)
            for env_id, config in environments.list().items()
        ],
        success=True,
    )


@app.get("/envs/{env_id}", response_model=EnvironmentSummary)
def get_environment(env_id: str) -> EnvironmentSummary:
    """Return a single environment config by env_id."""
    try:
        config = environments.get(env_id)
    except RegistryException as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return EnvironmentSummary(env_id=env_id, config=config)


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    """Train the experiment's agent and return the state-value function."""
    experiment = _get_experiment(req.experiment_id)

    try:
        params = dict(req.train_params or {})
        if "P" in params:
            params["P"] = normalize_transitions(params["P"])
        V = experiment.train(params)
        return TrainResponse(value=V.tolist(), success=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


def _predict_action(experiment_id: str, state: int) -> int:
    """Greedy action for the given state; raises HTTPException on failure."""
    experiment = _get_experiment(experiment_id)
    try:
        return experiment.predict(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Return the greedy action for the given state."""
    return PredictResponse(action=_predict_action(req.experiment_id, req.state))


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    """Persistent prediction channel for bulk evaluation.

    One connection is reused for many predict calls, avoiding the TCP/TLS and
    HTTP framing overhead of the REST endpoint. Each message is a JSON object:

      {"id": 1, "experiment_id": "...", "state": 3}
          -> {"id": 1, "action": 1}

      {"id": 2, "experiment_id": "...", "states": [0, 1, 2]}
          -> {"id": 2, "actions": [1, 1, 1]}

    Errors are reported back on the socket ({"id": ..., "error": "..."}) so the
    caller can keep using the same connection.
    """
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            msg_id = msg.get("id")
            experiment_id = msg.get("experiment_id")
            reply: dict[str, Any] = {"id": msg_id}
            try:
                experiment = _get_experiment(experiment_id)
                if "states" in msg:
                    states: list[Any] = msg["states"]
                    reply["actions"] = [
                        experiment.predict(int(s)) for s in states
                    ]
                elif "state" in msg:
                    reply["action"] = experiment.predict(int(msg["state"]))
                else:
                    raise ValueError("Message must contain 'state' or 'states'")
            except Exception as exc:
                if isinstance(exc, HTTPException):
                    reply["error"] = str(exc.detail)
                else:
                    reply["error"] = str(exc)
            await websocket.send_json(reply)
    except WebSocketDisconnect:
        pass


# ------------------------- Frontend static files -------------------------
# Served last so the JSON endpoints above always win. index.html is served
# at "/" thanks to html=True. Open http://127.0.0.1:8123/ to play.
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")