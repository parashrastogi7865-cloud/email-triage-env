"""
OpenEnv: Email Triage Environment
Real-world task: Classify, prioritize, and respond to emails
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional

# Try pydantic, fall back to dataclasses
try:
    from pydantic import BaseModel

    class EmailAction(BaseModel):
        email_id: str
        label: str
        priority: int
        reply: Optional[str] = None
        archive: bool = False

    class EmailObservation(BaseModel):
        inbox: list
        processed: list
        step_number: int
        remaining_emails: int
        last_action_result: Optional[str] = None
        def model_dump(self):
            return self.__dict__.copy()

    class EmailReward(BaseModel):
        total: float
        label_accuracy: float
        priority_accuracy: float
        reply_quality: float
        efficiency_bonus: float
        penalty: float
        def model_dump(self):
            return self.__dict__.copy()

except ImportError:
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
     "body":"The main prod DB has been unreachable for 10 minutes. Customers can't log in. Need immediate response.",
     "true_label":"urgent","true_priority":1,"needs_reply":True},
    {"id":"e002","from":"newsletter@techdigest.io","subject":"Top 10 AI trends this week",
     "body":"Hi there! Here are this week's top AI stories curated just for you...",
     "true_label":"promotional","true_priority":5,"needs_reply":False},
    {"id":"e003","from":"hr@company.com","subject":"Q2 performance review schedule",
     "body":"Please complete your self-evaluation by Friday. The review portal is now open.",
     "true_label":"normal","true_priority":3,"needs_reply":True},
    {"id":"e004","from":"winner@lottery-prize.biz","subject":"You've won $1,000,000!!!",
     "body":"Congratulations! You are selected. Click here to claim your prize immediately.",
     "true_label":"spam","true_priority":5,"needs_reply":False},
    {"id":"e005","from":"client@bigcorp.com","subject":"Contract renewal - deadline this week",
     "body":"Hi, our contract expires Friday. Please send the renewal docs ASAP or we'll go with another vendor.",
     "true_label":"urgent","true_priority":1,"needs_reply":True},
    {"id":"e006","from":"team@slack.com","subject":"Your Slack digest for today",
     "body":"Here's what happened in your Slack channels today: 3 new messages in #general...",
     "true_label":"promotional","true_priority":5,"needs_reply":False},
    {"id":"e007","from":"boss@company.com","subject":"Meeting notes from today",
     "body":"Attached are the notes from today's strategy meeting. Please review and add any corrections.",
     "true_label":"normal","true_priority":2,"needs_reply":True},
    {"id":"e008","from":"security@company.com","subject":"Suspicious login attempt detected",
     "body":"We detected a login attempt from IP 185.220.101.x (Tor exit node). Please verify immediately.",
     "true_label":"urgent","true_priority":1,"needs_reply":False},
]


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage Environment.
    API: reset() / step(action) / state()
    """

    metadata = {
        "name": "email-triage-v1",
        "version": "1.0.0",
        "description": "Real-world email triage: label, prioritize, and reply to emails",
        "action_space": "EmailAction",
        "observation_space": "EmailObservation",
        "reward_range": [0.0, 1.0],
    }

    def __init__(self, task_id: int = 0, seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._inbox = []
        self._processed = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_result = None

    def reset(self) -> EmailObservation:
        rng = random.Random(self.seed)
        emails = EMAILS.copy()
        rng.shuffle(emails)
        sizes = {0: 3, 1: 5, 2: 8}
        size = sizes.get(self.task_id, 8)
        self._inbox = [dict(e) for e in emails[:size]]
        self._processed = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_result = None
        return self._make_obs()

    def step(self, action: EmailAction):
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        target = next((e for e in self._inbox if e["id"] == action.email_id), None)
        if target is None:
            self._last_result = f"Email {action.email_id} not found."
            return self._make_obs(), 0.0, self._done, {"error": self._last_result}

        rew = self._score(action, target)
        self._inbox.remove(target)
        self._processed.append({**target,
            "agent_label": action.label, "agent_priority": action.priority,
            "agent_reply": action.reply, "agent_archived": action.archive,
            "reward": rew.total})
        self._step_count += 1
        self._cumulative_reward += rew.total
        self._last_result = f"Processed {action.email_id}: reward={rew.total:.3f}"
        self._done = len(self._inbox) == 0
        info = {"step": self._step_count, "step_reward": rew.total,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "reward_breakdown": rew.model_dump()}
        return self._make_obs(), rew.total, self._done, info

    def state(self) -> dict:
        return {"task_id": self.task_id, "step": self._step_count,
                "inbox_count": len(self._inbox), "processed_count": len(self._processed),
                "cumulative_reward": round(self._cumulative_reward, 4),
                "done": self._done, "inbox_ids": [e["id"] for e in self._inbox]}

    def _score(self, action: EmailAction, email: dict) -> EmailReward:
        label_s = 1.0 if action.label == email["true_label"] else 0.0

        diff = abs(action.priority - email["true_priority"])
        priority_s = 1.0 if diff == 0 else (0.5 if diff == 1 else max(0.0, 1.0 - diff * 0.25))

        if email["needs_reply"]:
            if action.reply and len(action.reply.strip()) > 20:
                reply_s = 1.0 if any(w in action.reply.lower()
                    for w in ["thank","apologi","will","please","regards"]) else 0.8
            elif action.reply:
                reply_s = 0.3
            else:
                reply_s = 0.0
        else:
            reply_s = -0.2 if (action.reply and len(action.reply.strip()) > 5) else 1.0

        should_arc = email["true_label"] in ("spam","promotional")
        eff_s = 1.0 if action.archive == should_arc else 0.0

        penalty = 0.0
        if email["true_label"] == "spam" and not action.archive:
            penalty += 0.2
        if email["true_label"] == "spam" and action.reply:
            penalty += 0.1

        raw = (0.35*label_s + 0.30*priority_s + 0.20*max(0.0,reply_s) + 0.15*eff_s - penalty)
        total = round(max(0.0, min(1.0, raw)), 4)
        return EmailReward(total=total, label_accuracy=label_s, priority_accuracy=priority_s,
                          reply_quality=reply_s, efficiency_bonus=eff_s, penalty=penalty)

    def _make_obs(self) -> EmailObservation:
        return EmailObservation(
            inbox=[{"id":e["id"],"from":e["from"],"subject":e["subject"],"body":e["body"]}
                   for e in self._inbox],
            processed=[{"id":e["id"],"label":e.get("agent_label"),"priority":e.get("agent_priority")}
                       for e in self._processed],
            step_number=self._step_count,
            remaining_emails=len(self._inbox),
            last_action_result=self._last_result)
