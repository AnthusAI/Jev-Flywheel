#!/usr/bin/env python
"""Plan or explicitly run a JEVPA live answer-cache top-up.

The default invocation is a no-spend plan. It never calls Jev. To execute, pass
``--spend``; the report then records actual request/token usage and estimated
cost using the caller-supplied rates. This is an answer collection scaffold,
not a complete preregistered arm runner: candidate selection and held-out
scoring remain separate steps.

Example plan (requires an explicit, non-test split):

    python scripts/run_jevpa_live.py --scorecard fixtures/scorecards/v1.yaml \
        --items fixtures/items.jsonl --split discovery --cache var/jevpa/live.jsonl \
        --input-usd-per-million 1 --output-usd-per-million 5 \
        --avg-input-tokens 500 --avg-output-tokens 80

Add ``--spend`` only after reviewing the JSON plan. A test/final split requires
``--final-evaluation`` and checksum-verified frozen winner scorecards. Token
rates and averages are estimates; actual usage is reported when returned by API.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from jev_flywheel.answers import AnswerCache, question_hash
from jev_flywheel.items import Item
from jev_flywheel.jev import JevSession
from jev_flywheel.scorecard import Scorecard

GLOBAL_REQUEST_CAP = 1820


def load_items(path: Path, split: str, *, final_evaluation: bool = False) -> list[Item]:
    """Load only items in the requested split; refuse held-out split names."""
    if split.strip().lower() in {"test", "final", "heldout", "held-out"} and not final_evaluation:
        raise ValueError("refusing to request answers for a held-out test/final split")
    items: list[Item] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = Item(**row)
            if item.split == split:
                items.append(item)
    if not items:
        raise ValueError(f"no items found for split {split!r} in {path}")
    return items


def load_manifest_items(path: Path, manifest_path: Path, fields: list[str], *,
                        final_evaluation: bool = False) -> list[Item]:
    """Load exactly the frozen IDs named by a split manifest, in manifest order."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    requested = []
    for field in fields:
        values = manifest.get(field)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"split manifest field {field!r} must be a list of IDs")
        requested.extend(values)
    if len(requested) != len(set(requested)):
        raise ValueError("selected manifest fields contain duplicate item IDs")
    if "test_ids" in fields and not final_evaluation:
        raise ValueError("refusing to request held-out test IDs outside final evaluation")
    source = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = Item(**json.loads(line))
                source[item.id] = item
    missing = [item_id for item_id in requested if item_id not in source]
    if missing:
        raise ValueError(f"{len(missing)} frozen IDs are absent from {path}")
    return [source[item_id] for item_id in requested]


def frozen_winner_questions(manifest_path: Path, *, request_cap: int = GLOBAL_REQUEST_CAP
                            ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load scorecards only when their frozen SHA-256 digests match the manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    winners = manifest.get("winners")
    if not isinstance(winners, list) or not winners:
        raise ValueError("winner manifest must contain a non-empty winners list")
    before = manifest.get("requests_used_before_final")
    if not isinstance(before, int) or before < 0:
        raise ValueError("winner manifest must record non-negative requests_used_before_final")
    if before > request_cap:
        raise ValueError("recorded pre-final requests exceed the registered request cap")
    merged_questions: dict[str, dict[str, Any]] = {}
    verified = []
    for winner in winners:
        card_path = (Path(manifest_path).parent / winner["scorecard"]).resolve()
        contents = card_path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        if digest != winner.get("sha256"):
            raise ValueError(f"frozen scorecard checksum mismatch for {winner.get('arm')!r}")
        card = Scorecard.from_yaml(contents.decode("utf-8"))
        for name, question in card.questions().items():
            existing = merged_questions.get(name)
            if existing is not None and existing != question:
                raise ValueError(f"winning scorecards redefine question {name!r}")
            merged_questions[name] = question
        verified.append({"arm": winner.get("arm"), "scorecard": str(card_path), "sha256": digest})
    return merged_questions, {"winners": verified, "requests_used_before_final": before,
                              "global_request_cap": request_cap,
                              "remaining_request_budget": request_cap - before}


def make_plan(items: list[Item], questions: dict[str, dict[str, Any]], cache: AnswerCache,
              *, split: str, model: str | None, input_rate: float | None,
              output_rate: float | None, avg_input_tokens: int | None,
              avg_output_tokens: int | None) -> dict[str, Any]:
    """Return an auditable estimate without initializing a Jev client."""
    plan = cache.plan([item.id for item in items], questions)
    prices_supplied = input_rate is not None and output_rate is not None
    averages_supplied = avg_input_tokens is not None and avg_output_tokens is not None
    estimated_input = plan.requests * avg_input_tokens if averages_supplied else None
    estimated_output = plan.requests * avg_output_tokens if averages_supplied else None
    estimated_cost = None
    if prices_supplied and averages_supplied:
        estimated_cost = (estimated_input * input_rate + estimated_output * output_rate) / 1_000_000
    return {
        "experiment": "JEVPA live answer collection scaffold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "plan_only",
        "spend_authorized": False,
        "split": split,
        "model": model,
        "item_count": len(items),
        "question_count": len(questions),
        "question_names": sorted(questions),
        "question_fingerprints": {name: question_hash(question)
                                   for name, question in sorted(questions.items())},
        "cache_path": str(cache.path) if cache.path else None,
        "cached_answers_before": len(cache),
        "planned_requests": plan.requests,
        "planned_missing_answers": plan.missing_answers,
        "items_needing_answers": plan.items_needing,
        "pricing": {
            "input_usd_per_million_tokens": input_rate,
            "output_usd_per_million_tokens": output_rate,
            "estimated_input_tokens_per_request": avg_input_tokens,
            "estimated_output_tokens_per_request": avg_output_tokens,
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimated_output,
            "estimated_cost_usd": estimated_cost,
            "estimate_available": estimated_cost is not None,
        },
        "actual": None,
        "note": "Plan only. No Jev client was initialized and no request was sent.",
    }


async def execute_plan(report: dict[str, Any], items: list[Item],
                       questions: dict[str, dict[str, Any]], cache: AnswerCache,
                       *, concurrency: int) -> dict[str, Any]:
    """Execute a previously displayed plan and append each successful answer."""
    session = JevSession()
    filled = await cache.fill(session, items, questions, concurrency=concurrency)
    report["mode"] = "live_execution"
    report["spend_authorized"] = True
    report["actual"] = asdict(filled)
    prices = report["pricing"]
    actual_cost = None
    if prices["input_usd_per_million_tokens"] is not None and prices["output_usd_per_million_tokens"] is not None:
        actual_cost = (filled.input_tokens * prices["input_usd_per_million_tokens"]
                       + filled.output_tokens * prices["output_usd_per_million_tokens"]) / 1_000_000
    report["actual"]["estimated_cost_usd"] = actual_cost
    report["actual"]["cached_answers_after"] = len(cache)
    report["note"] = "Live requests were sent. Successful answers were appended to the cache."
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_usd_per_million is not None and args.input_usd_per_million < 0:
        raise ValueError("input token price cannot be negative")
    if args.output_usd_per_million is not None and args.output_usd_per_million < 0:
        raise ValueError("output token price cannot be negative")
    if args.avg_input_tokens is not None and args.avg_input_tokens < 0:
        raise ValueError("average input tokens cannot be negative")
    if args.avg_output_tokens is not None and args.avg_output_tokens < 0:
        raise ValueError("average output tokens cannot be negative")
    if args.request_cap < 1:
        raise ValueError("request cap must be positive")
    manifest_info = None
    if args.final_evaluation:
        if not args.winner_manifest:
            raise ValueError("--final-evaluation requires --winner-manifest")
        questions, manifest_info = frozen_winner_questions(args.winner_manifest,
                                                            request_cap=args.request_cap)
        valid_final_split = (args.split is not None and args.split.lower() in {
            "test", "final", "heldout", "held-out"}) or (
                args.split_manifest is not None and args.manifest_fields == ["test_ids"])
        if not valid_final_split:
            raise ValueError("--final-evaluation requires an explicit test/final split")
    else:
        if args.winner_manifest:
            raise ValueError("--winner-manifest is only valid with --final-evaluation")
        if args.scorecard is None:
            raise ValueError("--scorecard is required outside final-evaluation mode")
        scorecard = Scorecard.from_yaml(args.scorecard.read_text(encoding="utf-8"))
        questions = scorecard.questions()
    if args.split_manifest:
        if args.split:
            raise ValueError("use either --split or --split-manifest, not both")
        items = load_manifest_items(args.items, args.split_manifest, args.manifest_fields,
                                    final_evaluation=args.final_evaluation)
        split_name = "+".join(args.manifest_fields)
    else:
        if not args.split:
            raise ValueError("--split is required without --split-manifest")
        items = load_items(args.items, args.split, final_evaluation=args.final_evaluation)
        split_name = args.split
    cache = AnswerCache(args.cache)
    report = make_plan(items, questions, cache, split=split_name, model=args.model,
                       input_rate=args.input_usd_per_million,
                       output_rate=args.output_usd_per_million,
                       avg_input_tokens=args.avg_input_tokens,
                       avg_output_tokens=args.avg_output_tokens)
    if manifest_info:
        report["final_evaluation"] = manifest_info
        if report["planned_requests"] > manifest_info["remaining_request_budget"]:
            raise ValueError("final batch exceeds the remaining global 1,820 request budget")
    elif report["planned_requests"] + args.requests_used > args.request_cap:
        raise ValueError("planned batch exceeds the registered request budget")
    already_used = (manifest_info["requests_used_before_final"] if manifest_info
                    else args.requests_used)
    report["global_request_budget"] = {
        "cap": args.request_cap,
        "used_before_this_batch": already_used,
        "planned_after_this_batch": already_used + report["planned_requests"],
    }
    if args.spend:
        load_dotenv()
        report = asyncio.run(execute_plan(report, items, questions, cache,
                                          concurrency=args.concurrency))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path)
    parser.add_argument("--final-evaluation", action="store_true",
                        help="Plan final test answers for checksum-verified frozen winners")
    parser.add_argument("--winner-manifest", type=Path,
                        help="JSON manifest with frozen winner scorecard checksums and prior request total")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--split", help="Explicit non-test corpus split to answer")
    parser.add_argument("--split-manifest", type=Path,
                        help="Frozen split JSON; use --manifest-fields to select exact IDs")
    parser.add_argument("--manifest-fields", nargs="+", default=[],
                        help="ID-list fields to load from --split-manifest")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", help="Recorded model label for the study plan")
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument("--avg-input-tokens", type=int)
    parser.add_argument("--avg-output-tokens", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-used", type=int, default=0,
                        help="Previously sent item-batch requests in this study")
    parser.add_argument("--request-cap", type=int, default=GLOBAL_REQUEST_CAP,
                        help="Pre-registered Jev request ceiling for this study")
    parser.add_argument("--spend", action="store_true",
                        help="Actually call Jev; omit for the default no-spend plan")
    parser.add_argument("--out", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.requests_used < 0:
        parser.error("--requests-used cannot be negative")
    if args.split_manifest and not args.manifest_fields:
        parser.error("--split-manifest requires --manifest-fields")
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
