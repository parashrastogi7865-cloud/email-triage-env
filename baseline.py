"""
Baseline inference script for Email Triage environment.
Uses OpenAI API client (reads OPENAI_API_KEY from environment).
Produces reproducible scores on all 3 tasks.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py
    OPENAI_API_KEY=sk-... python baseline.py --task 2
"""

import os
import json
import argparse
import textwrap
from openai import OpenAI

from env import EmailTriageEnv, EmailAction
from tasks import TASKS


SYSTEM_PROMPT = """You are an expert email triage assistant.
For each email, you must output a JSON object with these fields:
- email_id: the ID of the email
- label: one of "urgent", "normal", "spam", "promotional"
- priority: integer 1 (highest) to 5 (lowest)
- reply: string reply text if the email needs a reply, else null
- archive: true if the email should be archived (spam/promotional), else false

Respond ONLY with valid JSON. No markdown, no explanation."""


def build_user_prompt(observation) -> str:
    inbox_text = ""
    for email in observation.inbox:
        inbox_text += f"""
Email ID: {email['id']}
From: {email['from']}
Subject: {email['subject']}
Body: {email['body']}
---"""
    return f"""Triage the FIRST email in the inbox below. Output JSON for exactly one email.

INBOX:
{inbox_text}

Already processed: {[e['id'] for e in observation.processed]}
"""


def run_task(client: OpenAI, task_id: int, seed: int = 42) -> dict:
    task = TASKS[task_id]
    env = EmailTriageEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    
    total_reward = 0.0
    steps = 0

    print(f"\n{'='*50}")
    print(f"Task {task_id}: {task.name} [{task.difficulty.upper()}]")
    print(f"Description: {task.description}")
    print(f"Inbox size: {len(obs.inbox)} emails")
    print(f"{'='*50}")

    while not env._done and steps < 20:
        if not obs.inbox:
            break

        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            action = EmailAction(**parsed)
        except Exception as e:
            print(f"  Step {steps+1}: Parse error: {e}")
            # Fallback: act on first email with defaults
            first_id = obs.inbox[0]["id"]
            action = EmailAction(
                email_id=first_id,
                label="normal",
                priority=3,
                reply=None,
                archive=False,
            )

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        print(f"  Step {steps}: {action.email_id} → label={action.label}, "
              f"priority={action.priority}, reward={reward:.3f}")

    # Run grader
    processed_with_truth = []
    for e in env._processed:
        processed_with_truth.append(e)

    grade_result = task.grader(env._processed)
    avg_reward = total_reward / max(steps, 1)

    print(f"\nTask {task_id} Results:")
    print(f"  Steps: {steps}")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"  Grader Score: {grade_result['score']:.4f}")
    print(f"  Passed: {grade_result['passed']}")
    print(f"  Reason: {grade_result['reason']}")

    return {
        "task_id": task_id,
        "task_name": task.name,
        "difficulty": task.difficulty,
        "steps": steps,
        "avg_step_reward": round(avg_reward, 4),
        "grader_score": grade_result["score"],
        "passed": grade_result["passed"],
        "details": grade_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=None, help="Task ID (0, 1, 2). Default: all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)
    task_ids = [args.task] if args.task is not None else [0, 1, 2]

    results = []
    for tid in task_ids:
        result = run_task(client, tid, seed=args.seed)
        results.append(result)

    print(f"\n{'='*50}")
    print("BASELINE SUMMARY")
    print(f"{'='*50}")
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  Task {r['task_id']} ({r['difficulty']:8s}): "
              f"score={r['grader_score']:.3f} {status}")

    overall = sum(r["grader_score"] for r in results) / len(results)
    print(f"\n  Overall Average: {overall:.4f}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump({"seed": args.seed, "results": results, "overall": overall}, f, indent=2)
    print("\nSaved to baseline_results.json")


if __name__ == "__main__":
    main()
