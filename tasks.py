"""
Tasks and graders for Email Triage environment.
3 tasks: easy → medium → hard
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    id: int
    name: str
    description: str
    difficulty: str
    inbox_size: int
    grader: Callable


def grade_easy(processed: list[dict]) -> dict:
    """
    Easy: 3 emails (1 urgent, 1 promotional, 1 normal).
    Pass if agent correctly labels all 3 emails.
    Score 0.0–1.0 based on label accuracy.
    """
    if not processed:
        return {"score": 0.0, "passed": False, "reason": "No emails processed"}

    correct = sum(
        1 for e in processed
        if e.get("agent_label") == e.get("true_label")
    )
    score = round(correct / max(len(processed), 1), 3)
    passed = score >= 0.9
    return {
        "score": score,
        "passed": passed,
        "correct_labels": correct,
        "total": len(processed),
        "reason": f"{correct}/{len(processed)} labels correct",
    }


def grade_medium(processed: list[dict]) -> dict:
    """
    Medium: 5 emails. Agent must correctly label AND prioritize.
    Score weights: 60% label, 40% priority accuracy.
    """
    if not processed:
        return {"score": 0.0, "passed": False, "reason": "No emails processed"}

    label_scores = []
    priority_scores = []

    for e in processed:
        label_scores.append(1.0 if e.get("agent_label") == e.get("true_label") else 0.0)
        diff = abs((e.get("agent_priority") or 3) - e.get("true_priority", 3))
        priority_scores.append(max(0.0, 1.0 - diff * 0.3))

    label_avg = sum(label_scores) / len(label_scores)
    priority_avg = sum(priority_scores) / len(priority_scores)
    score = round(0.6 * label_avg + 0.4 * priority_avg, 3)
    passed = score >= 0.75

    return {
        "score": score,
        "passed": passed,
        "label_accuracy": round(label_avg, 3),
        "priority_accuracy": round(priority_avg, 3),
        "reason": f"Label={label_avg:.0%}, Priority={priority_avg:.0%}",
    }


def grade_hard(processed: list[dict]) -> dict:
    """
    Hard: 8 emails. Full triage: label + priority + reply + archive.
    Holistic score across all dimensions.
    """
    if not processed:
        return {"score": 0.0, "passed": False, "reason": "No emails processed"}

    scores = []
    reply_hits = 0
    archive_hits = 0

    for e in processed:
        label_ok = 1.0 if e.get("agent_label") == e.get("true_label") else 0.0
        diff = abs((e.get("agent_priority") or 3) - e.get("true_priority", 3))
        priority_ok = max(0.0, 1.0 - diff * 0.25)

        needs_reply = e.get("needs_reply", False)
        has_reply = bool(e.get("agent_reply") and len(str(e.get("agent_reply", "")).strip()) > 20)
        reply_ok = 1.0 if (needs_reply == has_reply) else 0.0
        if needs_reply and has_reply:
            reply_hits += 1

        should_archive = e.get("true_label") in ("spam", "promotional")
        archive_ok = 1.0 if (e.get("agent_archived") == should_archive) else 0.0
        if e.get("agent_archived") == should_archive:
            archive_hits += 1

        e_score = 0.35 * label_ok + 0.30 * priority_ok + 0.20 * reply_ok + 0.15 * archive_ok
        scores.append(e_score)

    total_score = round(sum(scores) / len(scores), 3)
    passed = total_score >= 0.70

    return {
        "score": total_score,
        "passed": passed,
        "emails_processed": len(processed),
        "reply_accuracy": f"{reply_hits}/{sum(1 for e in processed if e.get('needs_reply'))}",
        "archive_accuracy": f"{archive_hits}/{len(processed)}",
        "reason": f"Overall holistic score: {total_score:.1%}",
    }


TASKS = [
    Task(
        id=0,
        name="label-only-easy",
        description="Triage 3 emails by labeling them correctly (urgent/normal/spam/promotional).",
        difficulty="easy",
        inbox_size=3,
        grader=grade_easy,
    ),
    Task(
        id=1,
        name="label-and-priority-medium",
        description="Triage 5 emails: label correctly AND assign accurate priority (1=highest, 5=lowest).",
        difficulty="medium",
        inbox_size=5,
        grader=grade_medium,
    ),
    Task(
        id=2,
        name="full-triage-hard",
        description="Fully triage 8 emails: label, prioritize, reply where needed, archive appropriately.",
        difficulty="hard",
        inbox_size=8,
        grader=grade_hard,
    ),
]
