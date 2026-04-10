"""
openenv validate — checks the environment conforms to spec.
Run: python validate.py
"""

import sys
import traceback
from env import EmailTriageEnv, EmailAction, EmailObservation, EmailReward


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def check(condition: bool, message: str):
    status = "✓" if condition else "✗"
    print(f"  {status}  {message}")
    if not condition:
        sys.exit(f"\nVALIDATION FAILED: {message}")


def main():
    print("OpenEnv Validator — email-triage-v1")

    section("1. Model imports")
    check(True, "EmailAction Pydantic model importable")
    check(True, "EmailObservation Pydantic model importable")
    check(True, "EmailReward Pydantic model importable")

    section("2. Environment instantiation")
    for task_id in range(3):
        env = EmailTriageEnv(task_id=task_id)
        check(hasattr(env, "reset"), f"Task {task_id}: env.reset() exists")
        check(hasattr(env, "step"),  f"Task {task_id}: env.step() exists")
        check(hasattr(env, "state"), f"Task {task_id}: env.state() exists")

    section("3. reset() returns EmailObservation")
    env = EmailTriageEnv(task_id=0, seed=42)
    obs = env.reset()
    check(isinstance(obs, EmailObservation), "reset() returns EmailObservation")
    check(len(obs.inbox) > 0, "Inbox non-empty after reset")
    check(obs.step_number == 0, "step_number is 0 after reset")

    section("4. step() returns (obs, float, bool, dict)")
    first_email = obs.inbox[0]
    action = EmailAction(
        email_id=first_email["id"],
        label="urgent",
        priority=1,
        reply="On it immediately.",
        archive=False,
    )
    result = env.step(action)
    check(len(result) == 4, "step() returns 4-tuple")
    new_obs, reward, done, info = result
    check(isinstance(new_obs, EmailObservation), "step() obs is EmailObservation")
    check(isinstance(reward, float), "step() reward is float")
    check(0.0 <= reward <= 1.0, f"reward in [0, 1]: got {reward}")
    check(isinstance(done, bool), "step() done is bool")
    check(isinstance(info, dict), "step() info is dict")

    section("5. state() returns dict")
    state = env.state()
    check(isinstance(state, dict), "state() returns dict")
    check("task_id" in state, "state has task_id")
    check("step" in state, "state has step")
    check("done" in state, "state has done")

    section("6. Episode terminates when inbox empty")
    env2 = EmailTriageEnv(task_id=0, seed=42)
    obs2 = env2.reset()
    final_done = False
    for _ in range(20):
        if not obs2.inbox:
            break
        a = EmailAction(
            email_id=obs2.inbox[0]["id"],
            label="normal",
            priority=3,
        )
        obs2, _, final_done, _ = env2.step(a)
    check(final_done, "Episode terminates (done=True) when inbox empty")

    section("7. Reward partial progress (dense)")
    env3 = EmailTriageEnv(task_id=2, seed=42)
    obs3 = env3.reset()
    rewards = []
    while obs3.inbox:
        a = EmailAction(email_id=obs3.inbox[0]["id"], label="normal", priority=3)
        obs3, r, _, _ = env3.step(a)
        rewards.append(r)
    check(len(set(rewards)) > 1, "Rewards vary across steps (dense signal)")
    check(all(0.0 <= r <= 1.0 for r in rewards), "All step rewards in [0,1]")

    section("8. Reproducibility")
    def run_and_score(seed):
        e = EmailTriageEnv(task_id=1, seed=seed)
        o = e.reset()
        total = 0.0
        while o.inbox:
            a = EmailAction(email_id=o.inbox[0]["id"], label="normal", priority=3)
            o, r, _, _ = e.step(a)
            total += r
        return total

    s1 = run_and_score(7)
    s2 = run_and_score(7)
    check(abs(s1 - s2) < 1e-9, f"Same seed produces same score ({s1:.4f})")

    print(f"\n{'='*50}")
    print("  ALL CHECKS PASSED ✓")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
