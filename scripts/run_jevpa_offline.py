#!/usr/bin/env python
"""Run the preregistered cached-answer JEVPA search mechanics pilot offline.

The comparison uses seven pre-existing cached questions and scripted corpus labels.
It makes no Jev or analyst calls and does not use the network; local head fits are
part of the experiment and the result is exploratory by design.

    python scripts/run_jevpa_offline.py
    python scripts/run_jevpa_offline.py --fixtures fixtures --recording fixtures/recordings/simulated-labeler
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.answers import AnswerCache, import_answers_jsonl
from jev_flywheel.evaluate import summarize
from jev_flywheel.fit import build_training_set, fit_head, with_fit
from jev_flywheel.items import FeedbackItem, JsonlStore, agrees
from jev_flywheel.jevpa import Candidate, merge_question_sets, pareto_pool
from jev_flywheel.proposal import ElementAdd, Proposal, apply_proposal
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.scoring import predict


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _proposal_elements(reply: str) -> list[str]:
    """Extract stable question keys from a recorded analyst reply."""
    try:
        proposal = json.loads(reply)
    except (TypeError, json.JSONDecodeError):
        return []
    return sorted({str(item["key"]) for item in proposal.get("add_elements", [])
                   if isinstance(item, dict) and item.get("key")})


def feasibility_report(fixtures: Path, recording: Path) -> dict[str, Any]:
    """Summarize the recorded proposal and cached answers without scoring outcomes."""
    fixtures, recording = Path(fixtures), Path(recording)
    script = json.loads((recording / "script.json").read_text(encoding="utf-8"))
    proposals = []
    for step in script.get("steps", []):
        if step.get("op") == "steer" and step.get("analyst_reply"):
            proposals.extend(_proposal_elements(step["analyst_reply"]))

    feedback = list(_jsonl(recording / "feedback.jsonl"))
    label_count = sum(bool(row.get("final_answer_value")) for row in feedback)
    propensity_label_count = sum(
        bool(row.get("final_answer_value"))
        and (row.get("metadata") or {}).get("propensity") is not None
        for row in feedback
    )
    dataset_item_count = sum(1 for _ in _jsonl(fixtures / "items.jsonl"))

    coverage: dict[str, set[str]] = {f"{script.get('score', 'score').lower()}.{key}": set()
                                      for key in proposals}
    answer_file = recording / "extra_answers.jsonl.gz"
    if answer_file.exists():
        for row in _jsonl(answer_file):
            name = row.get("name")
            if name in coverage and row.get("item_id"):
                coverage[name].add(str(row["item_id"]))

    unique_candidate_sets = {tuple(proposals)} if proposals else set()
    can_compare = len(unique_candidate_sets) >= 3
    return {
        "experiment": "JEVPA offline sentiment fixture audit",
        "result_kind": "feasibility_only" if not proposals else "machinery_pilot_only",
        "network_calls": 0,
        "recorded_proposal_count": len(proposals),
        "distinct_candidate_sets": len(unique_candidate_sets),
        "three_arm_comparison_possible": can_compare,
        "dataset_items": dataset_item_count,
        "recorded_feedback_rows": len(feedback),
        "trusted_label_rows": label_count,
        "trusted_labels_with_propensity": propensity_label_count,
        "proposed_question_answer_coverage": {
            name: {"answered_items": len(item_ids), "dataset_items": dataset_item_count}
            for name, item_ids in sorted(coverage.items())
        },
        "interpretation": (
            "No recorded proposal is available; this is only a feasibility audit."
            if not proposals else
            "Only recorded question sets are available. Treat any replay as a machinery "
            "pilot; it cannot identify a Pareto-versus-greedy effect."
        ),
    }


def partition_pool_ids(rows: list[dict[str, Any]], discovery_ids: set[str], *, seed: int,
                       selection_size: int, test_size: int) -> tuple[list[str], list[str]]:
    """Make deterministic disjoint splits from the pool after discovery IDs are removed."""
    eligible = sorted(row["id"] for row in rows
                      if row.get("metadata", {}).get("split") == "pool"
                      and row["id"] not in discovery_ids)
    if len(eligible) < selection_size + test_size:
        raise ValueError(f"pool has {len(eligible)} remaining items; need "
                         f"{selection_size + test_size}")
    random.Random(seed).shuffle(eligible)
    return eligible[:selection_size], eligible[selection_size:selection_size + test_size]


def _single_scorecard(reference: Scorecard, element_key: str,
                      incumbent: Scorecard | None = None) -> Scorecard:
    """Create a candidate with one cached reference question and a fresh fitted head."""
    spec = next(item for item in reference.score("Sentiment").elements
                if item.key == element_key)
    incumbent = incumbent or Scorecard.from_yaml(
        Path("fixtures/scorecards/v1.yaml").read_text(encoding="utf-8"))
    candidate = apply_proposal(incumbent, "Sentiment", Proposal(add=[ElementAdd(
        key=spec.key, question_type=spec.question_type,
        instructions=spec.instructions, criteria=spec.criteria,
    )]))
    return candidate


def _measure(scorecard: Scorecard, item_ids: list[str], items: dict[str, dict],
             cache: AnswerCache) -> dict[str, Any]:
    score = scorecard.score("Sentiment")
    questions = scorecard.questions()
    confidences, correct = [], []
    missing = 0
    for item_id in item_ids:
        answers = cache.partial_answers_for(item_id, questions)
        if len(answers) != len(questions):
            missing += 1
            continue
        result = predict(score, answers)
        label = items[item_id]["metadata"]["reference_label"]
        confidences.append(float(result.confidence or 0.0))
        correct.append(int(agrees(result.value, label)))
    summary = summarize(confidences, correct)
    if missing:
        raise ValueError(f"{missing} of {len(item_ids)} evaluation items lack complete cached "
                         "answers; refusing to score a smaller sample")
    return {"n": summary.n, "accuracy": summary.accuracy, "brier": summary.brier,
            "ece": summary.ece, "missing_items": missing}


def _fit_candidate(candidate_id: str, scorecard: Scorecard, feedback: list[FeedbackItem],
                   cache: AnswerCache, selection_ids: list[str], items: dict[str, dict],
                   seed: int) -> tuple[Candidate, dict[str, Any]]:
    score = scorecard.score("Sentiment")
    training = build_training_set(score, scorecard.questions(), cache, feedback)
    fitted = fit_head(training, score, seed=seed)
    if not fitted.fitted:
        raise ValueError(f"{candidate_id}: fit held: {fitted.reason}")
    fitted_card = with_fit(scorecard, "Sentiment", fitted)
    overall = _measure(fitted_card, selection_ids, items, cache)
    slice_scores, slice_n = {}, {}
    for label in ("positive", "negative"):
        ids = [item_id for item_id in selection_ids
               if items[item_id]["metadata"]["reference_label"] == label]
        result = _measure(fitted_card, ids, items, cache)
        slice_scores[label] = result["brier"]
        # The selection sample is a simple random sample within this fixed slice.
        slice_n[label] = float(result["n"])
    record = {"candidate_id": candidate_id, "question_keys": [e.key for e in score.elements],
              "fit_n": fitted.n, "fit_n_effective": fitted.n_effective,
              "selection": overall, "selection_slices": slice_scores,
              "fit_status": fitted.status}
    return Candidate(candidate_id, fitted_card, slice_scores, slice_n,
                     overall["brier"]), record


def run_pilot(fixtures: Path = Path("fixtures"),
              recording: Path = Path("fixtures/recordings/simulated-labeler"), *,
              seed: int = 20260924, selection_size: int = 500,
              test_size: int = 600) -> dict[str, Any]:
    """Compare greedy and Pareto search, with the frozen incumbent as context."""
    fixtures, recording = Path(fixtures), Path(recording)
    rows = list(_jsonl(fixtures / "items.jsonl"))
    items = {row["id"]: row for row in rows}
    feedback = list(JsonlStore(recording / "feedback.jsonl", FeedbackItem))
    discovery_ids = {record.item_id for record in feedback if record.label is not None}
    selection_ids, test_ids = partition_pool_ids(rows, discovery_ids, seed=seed,
                                                 selection_size=selection_size,
                                                 test_size=test_size)
    reference = Scorecard.from_yaml(
        (fixtures / "scorecards" / "reference_full.yaml").read_text(encoding="utf-8"))
    incumbent = Scorecard.from_yaml(
        (fixtures / "scorecards" / "v1.yaml").read_text(encoding="utf-8"))
    questions = reference.questions()
    cache = AnswerCache()
    import_answers_jsonl(fixtures / "answers.jsonl.gz", questions, cache)
    discovery = [f for f in feedback if f.item_id in discovery_ids]

    singles: dict[str, Candidate] = {}
    candidate_records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for element in reference.score("Sentiment").elements:
        card = _single_scorecard(reference, element.key, incumbent)
        cid = f"single:{element.key}"
        try:
            candidate, record = _fit_candidate(cid, card, discovery, cache,
                                               selection_ids, items, seed)
            singles[element.key] = candidate
            candidate_records[cid] = record
        except Exception as error:  # Keep failures visible in the audit report.
            failures.append({"candidate_id": cid, "error": f"{type(error).__name__}: {error}"})
    if len(singles) != 7:
        raise RuntimeError(f"all seven singleton fits are required; completed {len(singles)}; "
                           f"failures={failures}")

    ordered = sorted(singles.values(), key=lambda c: (c.overall_brier, c.candidate_id))
    greedy_pairs = [(ordered[0], ordered[1]), (ordered[0], ordered[2])]
    frontier = pareto_pool(singles.values(), max_size=4, min_slice_effective_n=50)
    if len(frontier) < 2:
        raise RuntimeError("Pareto pool has fewer than two candidates for question-set merges")
    positive = sorted(frontier, key=lambda c: (c.slice_brier["positive"], c.candidate_id))
    negative = sorted(frontier, key=lambda c: (c.slice_brier["negative"], c.candidate_id))
    pareto_pairs: list[tuple[Candidate, Candidate]] = []
    pair_keys = set()
    # Prefer complementary slice winners; if a winner repeats, move down that slice's
    # deterministic ranking until a new compatible pair is found.
    for left in positive:
        for right in negative:
            if left.candidate_id == right.candidate_id:
                continue
            key = tuple(sorted((left.candidate_id, right.candidate_id)))
            if key not in pair_keys:
                pair_keys.add(key)
                pareto_pairs.append((left, right))
                break
        if len(pareto_pairs) == 2:
            break
    pareto_pair_fallback = len(pareto_pairs) < 2

    def fit_pairs(arm: str, pairs: list[tuple[Candidate, Candidate]]) -> list[Candidate]:
        fitted_pairs = []
        for index, (left, right) in enumerate(pairs, 1):
            cid = f"{arm}:merge-{index}:{left.candidate_id}+{right.candidate_id}"
            union = merge_question_sets(left.scorecard, right.scorecard, "Sentiment")
            candidate, record = _fit_candidate(cid, union, discovery, cache,
                                               selection_ids, items, seed)
            fitted_pairs.append(candidate)
            candidate_records[cid] = record
        return fitted_pairs

    greedy_merged = fit_pairs("greedy", greedy_pairs)
    pareto_merged = fit_pairs("jevpa", pareto_pairs)
    greedy_candidates = list(singles.values()) + greedy_merged
    pareto_candidates = list(singles.values()) + pareto_merged
    greedy_merge_sets = {
        tuple(sorted(candidate_records[cid]["question_keys"]))
        for cid in candidate_records if cid.startswith("greedy:merge-")
    }
    jevpa_merge_sets = [
        tuple(sorted(candidate_records[cid]["question_keys"]))
        for cid in candidate_records if cid.startswith("jevpa:merge-")
    ]
    greedy_winner = min(greedy_candidates, key=lambda c: (c.overall_brier, c.candidate_id))
    jevpa_winner = min(pareto_candidates, key=lambda c: (c.overall_brier, c.candidate_id))

    def arm_result(candidates: list[Candidate], winner: Candidate) -> dict[str, Any]:
        return {"candidate_fits": len(candidates), "winner": winner.candidate_id,
                "winner_question_keys": [e.key for e in winner.scorecard.score("Sentiment").elements],
                "winner_selection": candidate_records[winner.candidate_id]["selection"],
                "exploratory_test": _measure(winner.scorecard, test_ids, items, cache),
                "candidates": [c.candidate_id for c in candidates]}

    coverage = {}
    for key in [e.key for e in reference.score("Sentiment").elements]:
        name = f"sentiment.{key}"
        spec_question = questions[name]
        answered_selection = sum(cache.get(i, name, spec_question) is not None for i in selection_ids)
        answered_test = sum(cache.get(i, name, spec_question) is not None for i in test_ids)
        coverage[name] = {"selection_answered": answered_selection,
                          "selection_items": len(selection_ids), "test_answered": answered_test,
                          "test_items": len(test_ids)}
    return {
        "experiment": "JEVPA offline sentiment search mechanics pilot",
        "protocol": {"seed": seed, "discovery_items": len(discovery_ids),
                     "selection_items": len(selection_ids), "exploratory_test_items": len(test_ids),
                     "slices": ["positive", "negative"], "minimum_slice_n": 50,
                     "singleton_candidates": 7, "pair_fits_per_search_arm": 2},
        "evidence_status": "exploratory_scripted_reference_labels",
        "network_calls": 0, "analyst_calls": 0,
        "answer_coverage": coverage,
        "greedy": arm_result(greedy_candidates, greedy_winner),
        "jevpa": arm_result(pareto_candidates, jevpa_winner),
        "incumbent_context": {"candidate": "frozen_v1_holistic_only",
                              "exploratory_test": _measure(incumbent, test_ids, items, cache)},
        "pareto_pool": [{"candidate_id": c.candidate_id,
                         "question_keys": [e.key for e in c.scorecard.score("Sentiment").elements],
                         "selection_slices": dict(c.slice_brier)} for c in frontier],
        "pareto_pair_fallback_used": pareto_pair_fallback,
        "jevpa_budget_unused_reason": (
            "The bounded Pareto front contained only two candidates, yielding one unique "
            "question-set union; a second fit would duplicate that union."
            if pareto_pair_fallback else None
        ),
        "candidate_fits": candidate_records, "fit_failures": failures,
        "greedy_merge_question_sets": [list(keys) for keys in sorted(greedy_merge_sets)],
        "jevpa_merge_question_sets": [list(keys) for keys in jevpa_merge_sets],
        "merge_differs_from_greedy": any(keys not in greedy_merge_sets
                                         for keys in jevpa_merge_sets),
        "interpretation": (
            "This exercises cached-answer fitting, slice scoring, Pareto retention, and "
            "question-set merging. It cannot establish that JEVPA improves discovery or "
            "generalizes to live reviewer feedback."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    parser.add_argument("--recording", type=Path,
                        default=Path("fixtures/recordings/simulated-labeler"))
    parser.add_argument("--out", type=Path, help="Optional path for the JSON report.")
    parser.add_argument("--feasibility-only", action="store_true",
                        help="Report fixture coverage without fitting candidates.")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    report = (feasibility_report(args.fixtures, args.recording) if args.feasibility_only
              else run_pilot(args.fixtures, args.recording, seed=args.seed))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
