"""
app.py — FastAPI entry point for Email Triage OpenEnv.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from env import EmailTriageEnv, EmailAction
from tasks import TASKS

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

_envs: dict = {}


@app.post("/reset")
async def reset(request: Request):
    task_id, seed, session_id = 0, 42, "default"
    try:
        body = await request.json()
        if body and isinstance(body, dict):
            task_id = int(body.get("task_id", 0))
            seed = int(body.get("seed", 42))
            session_id = str(body.get("session_id", "default"))
    except Exception:
        pass
    env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    _envs[session_id] = env
    return JSONResponse(obs.model_dump())


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    session_id = body.get("session_id", "default")
    env = _envs.get(session_id)
    if env is None:
        return JSONResponse({"error": "Call /reset first"}, status_code=400)
    if env._done:
        return JSONResponse({"error": "Episode done. Call /reset."}, status_code=400)
    try:
        action = EmailAction(
            email_id=body["email_id"],
            label=body["label"],
            priority=int(body["priority"]),
            reply=body.get("reply"),
            archive=bool(body.get("archive", False)),
        )
    except Exception as e:
        return JSONResponse({"error": f"Invalid action: {e}"}, status_code=400)
    obs, reward, done, info = env.step(action)
    return JSONResponse({"observation": obs.model_dump(), "reward": reward,
                         "done": done, "info": info})


@app.get("/state")
async def state(session_id: str = "default"):
    env = _envs.get(session_id)
    if env is None:
        return JSONResponse({"error": "Call /reset first"}, status_code=400)
    return JSONResponse(env.state())


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


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()