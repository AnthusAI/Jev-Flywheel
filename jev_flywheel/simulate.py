"""A simulated labeler, for demos, tests and recordings.

It answers as the corpus's own reference label says, and explains disagreements with a fixed
template. That is enough to exercise the loop end to end, and it is *not* a human: it never
notices a factor nobody declared, so a run driven by it demonstrates the machinery and the
measurement, not the claim that human comments surface hidden factors. The recording made
from it says so in its README. The real claim needs real labels.
"""
import random
from typing import Callable, Optional

from jev_flywheel.items import normalize_label
from jev_flywheel.loop import AGREE, DISAGREE, next_question, record_label, refit, status
from jev_flywheel.workspace import Workspace


def template_comment(question, truth: str) -> str:
    tier = question.item.metadata.get("tier", "this")
    return f"This is really {truth}; the wording is {tier} and it misleads."


def label_with_reference(
    workspace: Workspace, score_name: str, count: int, *, seed: int = 0,
    comment: Optional[Callable] = template_comment, auto_refit: bool = True,
) -> int:
    """Label ``count`` questions using the corpus's reference labels.

    Refits whenever the steering policy says one is due, as the console does. Returns how
    many labels were given.
    """
    rng = random.Random(seed)
    for _ in range(count):
        question = next_question(workspace, score_name, rng)
        if question is None:
            break
        truth = question.item.reference_label
        if normalize_label(truth) == normalize_label(question.result.value):
            record_label(workspace, question, AGREE, editor="simulated-labeler")
        else:
            record_label(workspace, question, DISAGREE, correct_label=truth,
                         comment=comment(question, truth) if comment else None,
                         editor="simulated-labeler")
        if auto_refit and status(workspace, score_name).triggers["refit"].fire:
            refit(workspace, score_name)
    return count
