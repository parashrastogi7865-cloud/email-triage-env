"""
FastAPI wrapper for Email Triage OpenEnv.
POST /reset  — reset environment, returns observation
POST /step   — take action, returns (obs, reward, done, info)
GET  /state  — current internal state
GET  /health — health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env import EmailTriageEnv, EmailAction
from tasks import TASKS

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

_envs: dict = {}  # session_id -> env


class ResetRequest(BaseModel):
    task_id: int = 0
    seed: int = 42
    session_id: str = "default"


class StepRequest(BaseModel):
    email_id: str
    label: str
    priority: int
    reply: Optional[str] = None
    archive: bool = False
    session_id: str = "default"


@app.post("/reset")
def reset(req: ResetRequest):
    if req.task_id not in range(3):
        raise HTTPException(status_code=400, detail="task_id must be 0, 1, or 2")
    env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    _envs[req.session_id] = env
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")
    action = EmailAction(
        email_id=req.email_id,
        label=req.label,
        priority=req.priority,
        reply=req.reply,
        archive=req.archive,
    )
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}


@app.get("/state")
def state(session_id: str = "default"):
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()


@app.get("/tasks")
def list_tasks():
    return [{"id": t.id, "name": t.name, "difficulty": t.difficulty,
             "description": t.description, "inbox_size": t.inbox_size} for t in TASKS]


@app.get("/health")
def health():
    return {"status": "ok", "env": "email-triage-v1"}


@app.get("/")
def root():
    return {"env": "email-triage-v1", "version": "1.0.0",
            "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)