#!/usr/bin/env python
"""Build the committed fixtures from the public sentiment corpus.

This is a maintenance script, not part of the demo. It exists so the fixtures are
reproducible and their provenance is written down, and it is the only place that
knows where the source data lives.

    python scripts/build_fixtures.py \\
        --corpus ~/Projects/Jev-Calibration \\
        --answers path/to/element_answers.jsonl

Produces, under ``fixtures/``:

    items.jsonl          8,801 items: id, text, and metadata carrying the split,
                         the strength tier and the corpus's own reference label
    answers.jsonl.gz     Jev's cached answers, one line per item, in the extract
                         format ``AnswerCache`` imports

The corpus is the public dataset from AnthusAI/Jev-Calibration, so both files are
redistributable. Nothing here is client data.

Splits: the source project calls its fit split "calibration". Here that word
already means the confidence-calibration layer, so the split is renamed **pool**
(the items a human may be asked about and a head may be fit on) and the original
name is kept in ``metadata.source_split``. The **test** split is never shown to a
human and never touched by selection; it is the scoreboard.
"""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

TIER_FILES = {
    "strong_positive.txt": ("positive", "strong"),
    "strong_negative.txt": ("negative", "strong"),
    "medium_positive.txt": ("positive", "medium"),
    "medium_negative.txt": ("negative", "medium"),
    "weak_positive.txt": ("positive", "weak"),
    "weak_negative.txt": ("negative", "weak"),
    "neutral_positive.txt": ("positive", "neutral"),
    "neutral_negative.txt": ("negative", "neutral"),
}
SPLIT_NAMES = {"calibration": "pool", "test": "test"}


def example_id(text: str) -> str:
    # Must match the source project, because the cached answers are keyed by it.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_rows(corpus: Path):
    seen, rows = set(), []
    for filename, (label, tier) in TIER_FILES.items():
        for line in (corpus / "dataset" / filename).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            item_id = example_id(text)
            if item_id in seen:
                continue
            seen.add(item_id)
            rows.append({"id": item_id, "text": text, "label": label, "tier": tier})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="a checkout of AnthusAI/Jev-Calibration")
    parser.add_argument("--answers", type=Path, required=True,
                        help="per-item answer extract: one JSON object per line")
    parser.add_argument("--out", type=Path, default=Path("fixtures"))
    args = parser.parse_args()

    rows = load_rows(args.corpus)
    splits = json.loads((args.corpus / "data" / "splits.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    with (args.out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            source_split = splits[row["id"]]
            handle.write(json.dumps({
                "id": row["id"],
                "text": row["text"],
                "metadata": {
                    "split": SPLIT_NAMES[source_split],
                    "source_split": source_split,
                    "tier": row["tier"],
                    "reference_label": row["label"],
                },
            }, ensure_ascii=False) + "\n")

    known = {row["id"] for row in rows}
    kept = 0
    with args.answers.open("rb") as source, gzip.open(
            args.out / "answers.jsonl.gz", "wb", compresslevel=9) as target:
        for line in source:
            if json.loads(line)["id"] in known:
                target.write(line)
                kept += 1
    by_split = {}
    for row in rows:
        name = SPLIT_NAMES[splits[row["id"]]]
        by_split[name] = by_split.get(name, 0) + 1
    print(f"items: {len(rows)} {by_split}")
    print(f"answers: {kept} items, "
          f"{(args.out / 'answers.jsonl.gz').stat().st_size / 1e6:.2f} MB gzipped")
    shutil.copyfile(args.corpus / "LICENSE", args.out / "CORPUS_LICENSE") \
        if (args.corpus / "LICENSE").exists() else None


if __name__ == "__main__":
    main()
