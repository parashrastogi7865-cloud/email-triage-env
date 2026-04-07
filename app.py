"""
FastAPI wrapper for the Email Triage OpenEnv environment.
Deployed on Hugging Face Spaces.

Endpoints:
  POST /reset       - Reset environment
  POST /step        - Take a step
  GET  /state       - Get current state
  GET  /tasks       - List all tasks
  GET  /validate    - Run validation checks
"""

import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from env import EmailTriageEnv, EmailAction
from tasks import TASKS

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for AI agents",
    version="1.0.0",
)

# In-memory session store (single session for demo)
_env: Optional[EmailTriageEnv] = None


class ResetRequest(BaseModel):
    task_id: int = 0
    seed: int = 42


@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    if req.task_id not in range(3):
        raise HTTPException(400, "task_id must be 0, 1, or 2")
    _env = EmailTriageEnv(task_id=req.task_id, seed=req.seed)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: EmailAction):
    if _env is None:
        raise HTTPException(400, "Call /reset first")
    if _env._done:
        raise HTTPException(400, "Episode done. Call /reset to start a new one.")
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(400, "Call /reset first")
    return _env.state()


@app.get("/tasks")
def list_tasks():
    return [
        {
            "id": t.id,
            "name": t.name,
            "difficulty": t.difficulty,
            "description": t.description,
            "inbox_size": t.inbox_size,
        }
        for t in TASKS
    ]


@app.get("/validate")
def validate():
    result = subprocess.run(
        ["python", "validate.py"],
        capture_output=True, text=True
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


@app.get("/")
def root():
    return {
        "env": "email-triage-v1",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/validate"],
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
