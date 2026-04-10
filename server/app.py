"""
app.py — FastAPI entry point for Email Triage OpenEnv.
Fully self-contained: all environment logic inlined to avoid import failures.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# ── Inline environment (no external file imports needed) ──────────────────────

@dataclass
class EmailAction:
    email_id: str
    label: str
    priority: int
    reply: Optional[str] = None
    archive: bool = False

@dataclass
class EmailObservation:
    inbox: list
    processed: list
    step_number: int
    remaining_emails: int
    last_action_result: Optional[str] = None
    def model_dump(self):
        return {"inbox": self.inbox, "processed": self.processed,
                "step_number": self.step_number,
                "remaining_emails": self.remaining_emails,
                "last_action_result": self.last_action_result}

@dataclass
class EmailReward:
    total: float
    label_accuracy: float
    priority_accuracy: float
    reply_quality: float
    efficiency_bonus: float
    penalty: float
    def model_dump(self):
        return {k: getattr(self, k) for k in
                ["total","label_accuracy","priority_accuracy",
                 "reply_quality","efficiency_bonus","penalty"]}

EMAILS = [
    {"id":"e001","from":"cto@company.com","subject":"URGENT: Production database down",
     "body":"The main prod DB has been unreachable for 10 minutes. Customers can't log in.",
     "true_label":"urgent","true_priority":1,"needs_reply":True},
    {"id":"e002","from":"newsletter@techdigest.io","subject":"Top 10 AI trends this week",
     "body":"Here are this week's top AI stories curated just for you...",
     "true_label":"promotional","true_priority":5,"needs_reply":False},
    {"id":"e003","from":"hr@company.com","subject":"Q2 performance review schedule",
     "body":"Please complete your self-evaluation by Friday.",
     "true_label":"normal","true_priority":3,"needs_reply":True},
    {"id":"e004","from":"winner@lottery-prize.biz","subject":"You've won $1,000,000!!!",
     "body":"Congratulations! Click here to claim your prize immediately.",
     "true_label":"spam","true_priority":5,"needs_reply":False},
    {"id":"e005","from":"client@bigcorp.com","subject":"Contract renewal - deadline this week",
     "body":"Our contract expires Friday. Please send renewal docs ASAP.",
     "true_label":"urgent","true_priority":1,"needs_reply":True},
    {"id":"e006","from":"team@slack.com","subject":"Your Slack digest for today",
     "body":"Here's what happened in your Slack channels today...",
     "true_label":"promotional","true_priority":5,"needs_reply":False},
    {"id":"e007","from":"boss@company.com","subject":"Meeting notes from today",
     "body":"Attached are the notes from today's strategy meeting. Please review.",
     "true_label":"normal","true_priority":2,"needs_reply":True},
    {"id":"e008","from":"security@company.com","subject":"Suspicious login attempt detected",
     "body":"We detected a login from a Tor exit node. Please verify your account.",
     "true_label":"urgent","true_priority":1,"needs_reply":False},
]

TASKS_META = [
    {"id":0,"name":"label-only-easy","difficulty":"easy","description":"Label 3 emails correctly","inbox_size":3},
    {"id":1,"name":"label-and-priority-medium","difficulty":"medium","description":"Label and prioritize 5 emails","inbox_size":5},
    {"id":2,"name":"full-triage-hard","difficulty":"hard","description":"Full triage of 8 emails","inbox_size":8},
]

class EmailTriageEnv:
    metadata = {"name":"email-triage-v1","version":"1.0.0","reward_range":[0.0,1.0]}

    def __init__(self, task_id=0, seed=42):
        self.task_id = task_id
        self.seed = seed
        self._inbox = []
        self._processed = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_result = None

    def reset(self):
        rng = random.Random(self.seed)
        emails = EMAILS.copy()
        rng.shuffle(emails)
        size = {0:3,1:5,2:8}.get(self.task_id, 8)
        self._inbox = [dict(e) for e in emails[:size]]
        self._processed = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_result = None
        return self._make_obs()

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        target = next((e for e in self._inbox if e["id"] == action.email_id), None)
        if target is None:
            return self._make_obs(), 0.0, self._done, {"error": "email not found"}
        rew = self._score(action, target)
        self._inbox.remove(target)
        self._processed.append({**target,
            "agent_label":action.label,"agent_priority":action.priority,
            "agent_reply":action.reply,"agent_archived":action.archive,"reward":rew.total})
        self._step_count += 1
        self._cumulative_reward += rew.total
        self._last_result = f"Processed {action.email_id}: reward={rew.total:.3f}"
        self._done = len(self._inbox) == 0
        info = {"step":self._step_count,"step_reward":rew.total,
                "cumulative_reward":round(self._cumulative_reward,4),
                "reward_breakdown":rew.model_dump()}
        return self._make_obs(), rew.total, self._done, info

    def state(self):
        return {"task_id":self.task_id,"step":self._step_count,
                "inbox_count":len(self._inbox),"processed_count":len(self._processed),
                "cumulative_reward":round(self._cumulative_reward,4),
                "done":self._done,"inbox_ids":[e["id"] for e in self._inbox]}

    def _score(self, action, email):
        label_s = 1.0 if action.label == email["true_label"] else 0.0
        diff = abs(action.priority - email["true_priority"])
        priority_s = 1.0 if diff==0 else (0.5 if diff==1 else max(0.0,1.0-diff*0.25))
        if email["needs_reply"]:
            if action.reply and len(action.reply.strip()) > 20:
                reply_s = 1.0 if any(w in action.reply.lower()
                    for w in ["thank","apologi","will","please","regards"]) else 0.8
            elif action.reply:
                reply_s = 0.3
            else:
                reply_s = 0.0
        else:
            reply_s = -0.2 if (action.reply and len(action.reply.strip())>5) else 1.0
        should_arc = email["true_label"] in ("spam","promotional")
        eff_s = 1.0 if action.archive == should_arc else 0.0
        penalty = 0.2 if (email["true_label"]=="spam" and not action.archive) else 0.0
        raw = 0.35*label_s + 0.30*priority_s + 0.20*max(0.0,reply_s) + 0.15*eff_s - penalty
        return EmailReward(total=round(max(0.0,min(1.0,raw)),4),
            label_accuracy=label_s,priority_accuracy=priority_s,
            reply_quality=reply_s,efficiency_bonus=eff_s,penalty=penalty)

    def _make_obs(self):
        return EmailObservation(
            inbox=[{"id":e["id"],"from":e["from"],"subject":e["subject"],"body":e["body"]}
                   for e in self._inbox],
            processed=[{"id":e["id"],"label":e.get("agent_label"),"priority":e.get("agent_priority")}
                       for e in self._processed],
            step_number=self._step_count,
            remaining_emails=len(self._inbox),
            last_action_result=self._last_result)

# ── FastAPI app ───────────────────────────────────────────────────────────────

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
            email_id=body["email_id"], label=body["label"],
            priority=int(body["priority"]), reply=body.get("reply"),
            archive=bool(body.get("archive", False)))
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
    return TASKS_META


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