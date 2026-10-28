from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from exceptions.registry_execeptions import RegistryException
from experiments.run import Experiment
from registry.experiment_registry import ExperimentRegistry

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ------------------------- Request/Response Schemas -------------------------

class ExperimentRequest(BaseModel):
    agent_type: str
    experiment_name: str
    agent_params: dict = {}
    env_config: dict = {}


class TrainRequest(BaseModel):
    experiment_id: str
    train_params: dict = {}


class PredictRequest(BaseModel):
    experiment_id: str
    state: int


# In-memory registry of experiments, keyed by experiment id
experiments = ExperimentRegistry()


# ------------------------- JSON helpers -------------------------

def serialize_env(experiment: Experiment) -> dict:
    return experiment.env_config


# ------------------------- Endpoints -------------------------

@app.get("/health")
def health():
    """Return server liveness only."""
    return {"status": "alive"}


@app.post("/register_experiment")
def register_experiment(req: ExperimentRequest):
    """Pass the frontend-provided env descriptor through to the selected agent."""
    try:
        experiment = Experiment(
            agent_type=req.agent_type,
            experiment_name=req.experiment_name,
            agent_params=req.agent_params,
            env_config=req.env_config,
        )
        experiments.store_experiment(experiment.id, experiment)
        return {
            "success": True,
            "experiment_id": experiment.id,
            "env": serialize_env(experiment),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")


def _get_experiment(experiment_id: str) -> Experiment:
    try:
        return experiments.retrieve(experiment_id)
    except RegistryException as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/experiments")
def list_experiments():
    """List every registered experiment (session) from the registry."""
    sessions = [
        {
            "experiment_id": exp_id,
            "experiment_name": experiments.retrieve(exp_id).name,
            "trained": experiments.retrieve(exp_id).trained,
        }
        for exp_id in experiments.get_model_ids()
    ]
    return {"experiments": sessions, "success": True}


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    """Return a single registered experiment's details + env descriptor."""
    experiment = _get_experiment(experiment_id)
    return {
        "success": True,
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "trained": experiment.trained,
        "value": experiment.value_function.tolist()
        if experiment.value_function is not None else None,
        "policy": experiment.policy,
        "env": serialize_env(experiment),
    }


@app.post("/train")
def train(req: TrainRequest):
    """Train the experiment's agent and return the state-value function."""
    experiment = _get_experiment(req.experiment_id)

    try:
        print("Training parameters: ", req.train_params)
        V = experiment.train(req.train_params)
        return {"value": V.tolist(), "success": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@app.post("/predict")
def predict(req: PredictRequest):
    """Return the greedy action for the given state."""
    experiment = _get_experiment(req.experiment_id)

    try:
        action = experiment.predict(req.state)
        return {"action": action}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


# ------------------------- Frontend static files -------------------------
# Served last so the JSON endpoints above always win. index.html is served
# at "/" thanks to html=True. Open http://127.0.0.1:8123/ to play.
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
