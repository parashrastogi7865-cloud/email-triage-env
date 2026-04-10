---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 📬 Email Triage — OpenEnv Environment

A complete **real-world OpenEnv environment** where an AI agent learns to triage an email inbox — classifying, prioritizing, replying, and archiving emails just as a human knowledge worker would.

---

## 🌍 Environment Description

Email triage is a task performed by millions of professionals daily. The agent receives an inbox of realistic emails and must process each one by:

1. **Labeling** — `urgent` / `normal` / `spam` / `promotional`
2. **Prioritizing** — Integer 1 (critical) to 5 (ignore)
3. **Replying** — When appropriate, write a professional reply
4. **Archiving** — Remove spam/promotional from the active inbox

The environment provides dense reward signals throughout the episode, not just at the end.

---

## 🔧 Action Space

```python
class EmailAction(BaseModel):
    email_id: str           # ID of email to act on
    label: str              # "urgent" | "normal" | "spam" | "promotional"
    priority: int           # 1 (highest) to 5 (lowest)
    reply: Optional[str]    # Reply text if needed, else None
    archive: bool           # True to archive
```

## 👁️ Observation Space

```python
class EmailObservation(BaseModel):
    inbox: list[dict]           # Unprocessed emails (id, from, subject, body)
    processed: list[dict]       # Already processed emails
    step_number: int
    remaining_emails: int
    last_action_result: Optional[str]
```

## 🎯 Reward Function

| Component         | Weight | Description |
|-------------------|--------|-------------|
| Label Accuracy    | 35%    | Correct classification |
| Priority Accuracy | 30%    | Within ±1 gets partial credit |
| Reply Quality     | 20%    | Reply when needed; don't reply to spam |
| Archive Efficiency| 15%    | Archive spam/promo; keep urgent/normal |

**Penalties:**
- Not archiving spam: −0.20
- Replying to spam: −0.10

All rewards are in range `[0.0, 1.0]` per step.

---

## 📋 Tasks

| ID | Name | Difficulty | Emails | Pass Threshold |
|----|------|-----------|--------|---------------|
| 0  | `label-only-easy` | Easy | 3 | ≥ 90% label accuracy |
| 1  | `label-and-priority-medium` | Medium | 5 | ≥ 75% weighted score |
| 2  | `full-triage-hard` | Hard | 8 | ≥ 70% holistic score |

---

## 🚀 Quick Start

### Install

```bash
git clone https://huggingface.co/spaces/yourname/email-triage-openenv
cd email-triage-openenv
pip install -r requirements.txt
```

### Run Validation

```bash
python validate.py
```

### Python API

```python
from env import EmailTriageEnv, EmailAction

env = EmailTriageEnv(task_id=1, seed=42)
obs = env.reset()

while obs.remaining_emails > 0:
    email = obs.inbox[0]
    action = EmailAction(
        email_id=email["id"],
        label="urgent",
        priority=1,
        reply="I'll look into this right away.",
        archive=False,
    )
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.3f} | Done: {done}")

print(env.state())
```

### Baseline (requires OpenAI API key)

```bash
OPENAI_API_KEY=sk-... python baseline.py
# Run specific task:
OPENAI_API_KEY=sk-... python baseline.py --task 2
```

### REST API (Hugging Face Spaces)

```bash
python app.py  # runs on :7860
```

```
POST /reset      {"task_id": 0, "seed": 42}
POST /step       {"email_id": "e001", "label": "urgent", "priority": 1, "archive": false}
GET  /state
GET  /tasks
GET  /validate
```

---

## 🐳 Docker

```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 email-triage-openenv
```
VERSIONS
openenv-core>=0.2.0
pydantic>=2.0
fastapi>=0.110.0
uvicorn>=0.27.0
openai>=1.0
pyyaml>=6.0
---

## 📁 File Structure

```
email-triage-openenv/
├── env.py           # Core environment (step/reset/state + reward)
├── tasks.py         # 3 tasks with programmatic graders
├── validate.py      # OpenEnv spec compliance validator
├── baseline.py      # Baseline agent using OpenAI API
├── app.py           # FastAPI REST wrapper for HF Spaces
├── openenv.yaml     # Environment metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Baseline Results (GPT-4o-mini, seed=42)

| Task | Difficulty | Score | Pass |
|------|-----------|-------|------|
| 0    | Easy      | ~0.93 | ✓    |
| 1    | Medium    | ~0.81 | ✓    |
| 2    | Hard      | ~0.72 | ✓    |

---

## 📜 License

MIT