"""
inference.py — Baseline inference script for Email Triage OpenEnv.
Required by the OpenEnv hackathon checker at repo root.

Runs a simple rule-based agent against all 3 tasks and prints scores.
For LLM-based inference, set OPENAI_API_KEY and pass --llm flag.

Usage:
    python inference.py                        # rule-based agent
    OPENAI_API_KEY=sk-... python inference.py --llm
"""

import os
import json
import argparse

from env import EmailTriageEnv, EmailAction
from tasks import TASKS


# ── Rule-based baseline agent ─────────────────────────────────────────────────

SPAM_KEYWORDS = ["won", "prize", "lottery", "congratulations", "click here", "claim"]
URGENT_KEYWORDS = ["urgent", "asap", "immediately", "down", "critical", "deadline", "expires"]
PROMO_SENDERS = ["newsletter", "digest", "noreply", "no-reply", "team@slack", "notify"]


def rule_based_agent(email: dict) -> EmailAction:
    subject = email["subject"].lower()
    body = email["body"].lower()
    sender = email["from"].lower()
    text = subject + " " + body

    # Classify
    if any(k in text for k in SPAM_KEYWORDS) or ".biz" in sender:
        label = "spam"
        priority = 5
        reply = None
        archive = True
    elif any(k in sender for k in PROMO_SENDERS):
        label = "promotional"
        priority = 5
        reply = None
        archive = True
    elif any(k in text for k in URGENT_KEYWORDS):
        label = "urgent"
        priority = 1
        reply = "Thank you for flagging this. I will address it immediately and keep you updated."
        archive = False
    else:
        label = "normal"
        priority = 3
        reply = "Thank you for your email. I will review this and get back to you shortly."
        archive = False

    return EmailAction(
        email_id=email["id"],
        label=label,
        priority=priority,
        reply=reply,
        archive=archive,
    )


# ── LLM agent (optional) ──────────────────────────────────────────────────────

def llm_agent(email: dict, client, model: str = "gpt-4o-mini") -> EmailAction:
    import json as _json
    prompt = f"""Triage this email and return ONLY a JSON object with fields:
email_id, label (urgent/normal/spam/promotional), priority (1-5),
reply (string or null), archive (true/false).

From: {email['from']}
Subject: {email['subject']}
Body: {email['body']}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an email triage assistant. Respond only with JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = _json.loads(raw)
    data["email_id"] = email["id"]  # ensure correct ID
    return EmailAction(**data)


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: int, agent_fn, seed: int = 42) -> dict:
    task = TASKS[task_id]
    env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    total_reward = 0.0
    steps = 0

    print(f"\n{'='*55}")
    print(f"Task {task_id}: {task.name}  [{task.difficulty.upper()}]")
    print(f"Inbox: {len(obs.inbox)} emails")
    print(f"{'='*55}")

    while obs.remaining_emails > 0 and steps < 25:
        email = obs.inbox[0]
        action = agent_fn(email)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"  [{steps}] {action.email_id:5s} → {action.label:12s} "
              f"pri={action.priority}  reward={reward:.3f}")
        if done:
            break

    grade = task.grader(env._processed)
    avg_reward = total_reward / max(steps, 1)

    print(f"\n  Avg reward : {avg_reward:.4f}")
    print(f"  Grade score: {grade['score']:.4f}")
    print(f"  Passed     : {'✓' if grade['passed'] else '✗'}  ({grade['reason']})")

    return {
        "task_id": task_id,
        "task_name": task.name,
        "difficulty": task.difficulty,
        "steps": steps,
        "avg_reward": round(avg_reward, 4),
        "score": grade["score"],
        "passed": grade["passed"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=None, help="Task 0/1/2 (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm", action="store_true", help="Use OpenAI LLM agent")
    args = parser.parse_args()

    if args.llm:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY to use --llm")
        client = OpenAI(api_key=api_key)
        agent_fn = lambda email: llm_agent(email, client)
        print("Agent: GPT-4o-mini (LLM)")
    else:
        agent_fn = rule_based_agent
        print("Agent: Rule-based baseline")

    task_ids = [args.task] if args.task is not None else [0, 1, 2]
    results = []
    for tid in task_ids:
        results.append(run_task(tid, agent_fn, seed=args.seed))

    print(f"\n{'='*55}")
    print("FINAL SCORES")
    print(f"{'='*55}")
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  Task {r['task_id']} ({r['difficulty']:8s}): "
              f"score={r['score']:.3f}  {status}")

    overall = sum(r["score"] for r in results) / len(results)
    print(f"\n  Overall: {overall:.4f}")

    with open("inference_results.json", "w") as f:
        json.dump({"seed": args.seed, "results": results, "overall": overall}, f, indent=2)
    print("  Saved → inference_results.json")


if __name__ == "__main__":
    main()
