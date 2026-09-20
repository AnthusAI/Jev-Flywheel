"""The on-disk answer cache, keyed per question rather than per question set.

This is the economic center of the project. The runtime session in ``jev.py``
fingerprints the *whole* question set, so adding one element voids every answer
it holds. That is correct for a runtime cache: the answers on hand came from a
different request. It would be ruinous here, where the entire point is proposing
new elements, because each proposal would cost a full Jev pass over every
labeled item.

So this cache is keyed by ``(item_id, question name, hash of the question body)``.
A newly proposed element then costs one top-up request per item that asks *only*
the new question, an element whose wording changed is re-asked (its body hash
changed) while every other answer is reused, and a retired element's answers
simply stop being read.

Only ``item_id`` is stored, never the text, so this file is not a second copy of
the corpus and can be committed as a fixture without republishing it.

Whether an answer depends on which other questions rode in the same request is an
open question (the holistic answer moved on 1.26% of items against about 1%
run-to-run noise, so the effect is small if it exists). If it turns out to be
material for elements, this cache has to key on the set fingerprint too and the
economics need re-deriving. ``tests/`` should grow a live measurement of it.
"""
from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from jev_flywheel.items import Item
from jev_flywheel.jev import JevSession, fingerprint, normalize_answer

Key = Tuple[str, str, str]  # (item_id, question name, question body hash)


def question_hash(question: Mapping[str, Any]) -> str:
    """Identifies one question's exact wording, type and criteria."""
    return fingerprint(dict(question))


@dataclass
class Plan:
    """What a top-up would cost, computed without spending anything.

    ``requests`` is the number of Jev calls, which is the number of items with at
    least one missing answer -- *not* the number of missing answers. All of an
    item's missing questions ride in one request, so ten new elements cost the
    same number of requests as one. That is why a cycle must batch every
    proposal into a single question-set change.
    """

    requests: int
    missing_answers: int
    items_needing: List[str] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        return self.requests == 0


@dataclass
class FillReport:
    requested: int = 0
    already_complete: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # A few distinct failure messages, so "all 140 requests failed" can say why.
    errors: List[str] = field(default_factory=list)


class AnswerCache:
    """Append-only JSONL of individual answers, indexed in memory."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._answers: Dict[Key, dict] = {}
        self.model: Optional[str] = None
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["item_id"], row["name"], row["qhash"])
                self._answers[key] = row["answer"]
                self.model = row.get("model") or self.model

    def __len__(self) -> int:
        return len(self._answers)

    def put(self, item_id: str, name: str, question: Mapping[str, Any],
            answer: Mapping[str, Any], model: Optional[str] = None) -> None:
        answer = normalize_answer(answer)
        qhash = question_hash(question)
        self._answers[(item_id, name, qhash)] = answer
        self.model = model or self.model
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {"item_id": item_id, "name": name, "qhash": qhash,
                   "model": model, "answer": answer}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    def put_hashed(self, item_id: str, name: str, qhash: str, answer: Mapping[str, Any],
                   model: Optional[str] = None) -> None:
        """Store an answer under a question hash that is already known.

        Used to restore a recording: its rows carry the hash of the wording they were
        collected under, and that wording is not needed to put them back.
        """
        answer = normalize_answer(answer)
        self._answers[(item_id, name, qhash)] = answer
        self.model = model or self.model
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {"item_id": item_id, "name": name, "qhash": qhash,
                   "model": model, "answer": answer}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    def rows(self):
        """Every stored answer as ``(item_id, name, qhash, answer)``."""
        for (item_id, name, qhash), answer in self._answers.items():
            yield item_id, name, qhash, answer

    def get(self, item_id: str, name: str, question: Mapping[str, Any]) -> Optional[dict]:
        return self._answers.get((item_id, name, question_hash(question)))

    def missing(self, item_id: str, questions: Mapping[str, Mapping[str, Any]]) -> Dict[str, dict]:
        """The questions this item has no answer on file for, at their current wording."""
        return {name: dict(q) for name, q in questions.items()
                if self.get(item_id, name, q) is None}

    def answers_for(self, item_id: str,
                    questions: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, dict]]:
        """Every answer for the current question set, or None if any is missing.

        Returning None rather than a partial mapping is deliberate. A caller that
        wants a partial vector to serve from can ask for it explicitly; a caller
        that is fitting must not be handed one by accident.
        """
        found: Dict[str, dict] = {}
        for name, question in questions.items():
            answer = self.get(item_id, name, question)
            if answer is None:
                return None
            found[name] = answer
        return found

    def partial_answers_for(self, item_id: str,
                            questions: Mapping[str, Mapping[str, Any]]) -> Dict[str, dict]:
        """Whatever answers are on file, for serving with reduced coverage."""
        found: Dict[str, dict] = {}
        for name, question in questions.items():
            answer = self.get(item_id, name, question)
            if answer is not None:
                found[name] = answer
        return found

    def bulk_partial_answers(
        self, item_ids: Iterable[str], questions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, dict]]:
        """``partial_answers_for`` over many items, hashing each question only once.

        Scoring the whole pool for every label would otherwise re-hash every
        question body tens of thousands of times.
        """
        hashes = {name: question_hash(q) for name, q in questions.items()}
        out: Dict[str, Dict[str, dict]] = {}
        for item_id in item_ids:
            found: Dict[str, dict] = {}
            for name, qhash in hashes.items():
                answer = self._answers.get((item_id, name, qhash))
                if answer is not None:
                    found[name] = answer
            out[item_id] = found
        return out

    def plan(self, item_ids: Iterable[str],
             questions: Mapping[str, Mapping[str, Any]]) -> Plan:
        """Price a top-up before paying for it."""
        needing: List[str] = []
        missing_answers = 0
        for item_id in item_ids:
            gap = self.missing(item_id, questions)
            if gap:
                needing.append(item_id)
                missing_answers += len(gap)
        return Plan(requests=len(needing), missing_answers=missing_answers,
                    items_needing=needing)

    async def fill(
        self,
        session: JevSession,
        items: Iterable[Item],
        questions: Mapping[str, Mapping[str, Any]],
        *,
        concurrency: int = 16,
        on_progress: Optional[Callable[[int, int], Awaitable[None] | None]] = None,
    ) -> FillReport:
        """Ask Jev only for what is missing, one request per item that needs anything.

        Failures are counted and never raised, and every completed answer is
        written immediately, so a run that dies partway resumes where it stopped
        instead of starting over.
        """
        report = FillReport()
        todo: List[Tuple[Item, Dict[str, dict]]] = []
        for item in items:
            gap = self.missing(item.id, questions)
            if gap:
                todo.append((item, gap))
            else:
                report.already_complete += 1
        total = len(todo)
        semaphore = asyncio.Semaphore(concurrency)
        done = 0

        async def one(item: Item, gap: Dict[str, dict]) -> None:
            nonlocal done
            async with semaphore:
                try:
                    result = await session.ask(item.text, gap)
                except Exception as error:  # noqa: BLE001 - one bad item must not sink the run
                    report.failures += 1
                    message = f"{type(error).__name__}: {error}"[:200]
                    if message not in report.errors and len(report.errors) < 3:
                        report.errors.append(message)
                else:
                    for name, answer in result.answers.items():
                        if name in gap:
                            self.put(item.id, name, gap[name], answer, result.model)
                    report.requested += 1
                    usage = result.usage or {}
                    report.input_tokens += usage.get("input_tokens") or 0
                    report.output_tokens += usage.get("output_tokens") or 0
                done += 1
                if on_progress is not None:
                    maybe = on_progress(done, total)
                    if asyncio.iscoroutine(maybe):
                        await maybe

        await asyncio.gather(*(one(item, gap) for item, gap in todo))
        return report


def import_answers_jsonl(
    source: Path,
    questions: Mapping[str, Mapping[str, Any]],
    cache: AnswerCache,
) -> int:
    """Load a per-item extract, ``{"id", "model", "answers": {name: answer}}`` per line.

    This is the format the Plexus experiment scripts wrote. Only names present in
    ``questions`` are imported, and each is stored under the hash of its *current*
    body, so an extract taken under different wording is correctly treated as
    stale rather than silently reused. Returns how many answers were stored.

    The extract is trusted to have been asked with exactly these question bodies.
    That holds for the bundled fixture, which was produced from the same
    definitions; for anything else, re-ask rather than import.
    """
    stored = 0
    source = Path(source)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for name, answer in (row.get("answers") or {}).items():
                if name in questions:
                    cache.put(row["id"], name, questions[name], answer, row.get("model"))
                    stored += 1
    return stored
