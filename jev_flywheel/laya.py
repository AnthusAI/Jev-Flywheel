"""Laya as a second engine: the same questions, answered locally.

Laya (https://huggingface.co/convaiinnovations/laya, Apache 2.0) is an encoder that answers
the same three typed questions Jev does -- ``noul``, ``choice``, ``score`` -- in one forward
pass, on your machine. This module runs it through ``laya-mlx``, an *independent* Apple-silicon
port (not an official Convai Innovations release).

The seam is deliberately thin. ``Agent.system_one(state, questions)`` already returns Jev's
response shape, so ``LayaClient`` only has to look like the SDK client and
``JevSession(client_factory=LayaClient)`` runs the whole flywheel unchanged: one request per
item, in-flight dedupe, counters, the per-question answer cache. Nothing above this file knows
which engine answered.

Three things differ from Jev, and they are why this is a module and not a config flag:

* **Every question is its own row.** Laya encodes the state once *per question*, so the cost of
  a request is proportional to the number of questions (Jev's is dominated by the item text).
  ``usage.input_tokens`` reports that honestly, as a sum over rows.
* **It truncates silently.** The state is cut to fit the 512-token window, each option to 48
  tokens, and the instruction head to what is left, with no error. An answer computed on half
  an item looks exactly like one computed on all of it, so this module counts first and refuses.
* **It is a small, weak-zero-shot model that ships over-confident.** That is the point of the
  experiment, not a defect to hide: the layer above it is what the repo demonstrates.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

DEFAULT_CHECKPOINT = "aac6fef/laya-mlx"
# Laya reports one option's tokens up to this many and drops the rest.
OPTION_TOKEN_LIMIT = 48
# Its own benchmark: Banking77 (77 options) 0.425 against Jev's 0.870. Not enforced by Laya.
MAX_CHOICE_OPTIONS = 20


class LayaError(RuntimeError):
    """The request cannot be answered faithfully by Laya."""


class LayaBudgetError(LayaError):
    """The question or state would be truncated. Says by how much, and where."""


@dataclass
class LayaStats:
    """What a local engine costs, in the units it actually spends."""

    calls: int = 0
    rows: int = 0            # question rows encoded; state tokens are paid once per row
    seconds: float = 0.0
    input_tokens: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def record(self, rows: int, seconds: float, tokens: int) -> None:
        self.calls += 1
        self.rows += rows
        self.seconds += seconds
        self.input_tokens += tokens
        self.latencies_ms.append(seconds * 1000.0)


def to_laya_state(state: Any) -> Any:
    """Unwrap Jev's ``{"text": ...}`` envelope.

    Laya JSON-encodes a dict state, so the wrapper would spend tokens and put ``{"text":`` in
    front of every item. Jev needs the envelope; Laya wants the text.
    """
    if isinstance(state, Mapping) and set(state) == {"text"}:
        return state["text"]
    return state


def to_laya_question(question: Mapping[str, Any]) -> Dict[str, Any]:
    """Our wire question to Laya's.

    Choice criteria pass through (Laya accepts a dict of label to description, with ``None``
    for none). Score criteria must be an ordered *list*; a dict is read as its keys, in order.
    Noul criteria, if any, must be a ``{"false":..., "true":...}`` dict.
    """
    out = {"type": question["type"], "instructions": question.get("instructions", "")}
    criteria = question.get("criteria")
    if question["type"] == "score" and isinstance(criteria, Mapping):
        criteria = list(criteria)
    if criteria:
        out["criteria"] = criteria
    return out


def check_budget(agent: Any, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse anything Laya would silently truncate. Raises ``LayaBudgetError``.

    Mirrors ``laya_mlx.common.build_sequence`` exactly, because a guard that is only
    approximately Laya's own arithmetic passes exactly the borderline requests it exists for.
    Leans on two of laya-mlx's helpers (``build_prefix``, ``render_options``) and its
    ``_to_internal``, so the extra pins ``laya-mlx==0.1.0``.
    """
    from laya_mlx.common import build_prefix, render_options, serialize_state

    tok = agent.tok
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "),
                    add_special_tokens=False)["input_ids"]

    for name, definition in questions.items():
        if definition.get("type") == "choice":
            n = len(definition.get("criteria") or ())
            if n > MAX_CHOICE_OPTIONS:
                raise LayaError(
                    f"question {name!r} has {n} options; Laya is weak past "
                    f"{MAX_CHOICE_OPTIONS} (77 options scored 0.425 in its own benchmark)")
        q = agent._to_internal(definition)
        for i, option in enumerate(render_options(q)):
            used = len(tok(" " + option.replace(tok.mask_token, " "),
                           add_special_tokens=False)["input_ids"])
            if used > OPTION_TOKEN_LIMIT:
                raise LayaBudgetError(
                    f"question {name!r} option {i} is {used} tokens; Laya keeps "
                    f"{OPTION_TOKEN_LIMIT} and drops the rest")
        ids, markers = build_prefix(tok, q, head_max_len)
        if len(markers) != len(render_options(q)):
            raise LayaBudgetError(
                f"question {name!r} has too many options for the {head_max_len}-token head")
        wanted = len(tok("%s question: %s" % (q["t"], str(q["ins"]).replace(tok.mask_token, " ")),
                         add_special_tokens=False)["input_ids"])
        # ids is [CLS] head [SEP] (mask + option)... [SEP]; what the head kept is what is left
        # once the options and the three specials are taken out.
        options = sum(1 + min(OPTION_TOKEN_LIMIT, len(tok(
            " " + o.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]))
            for o in render_options(q))
        kept = len(ids) - 3 - options
        if wanted > kept:
            raise LayaBudgetError(
                f"question {name!r} instructions are {wanted} tokens but Laya keeps "
                f"{max(kept, 0)} beside {len(markers)} options in its {head_max_len}-token head")
        room = max(0, max_len - len(ids) - 1)
        if len(state_ids) > room:
            raise LayaBudgetError(
                f"state is {len(state_ids)} tokens but only {room} are available after "
                f"question {name!r} ({len(ids)} tokens of instructions and options, "
                f"{max_len}-token window); Laya would silently drop the last "
                f"{len(state_ids) - room}")


class LayaClient:
    """Looks like the TypeSafe SDK's async client, backed by a local Laya checkpoint.

    Inference runs on one dedicated worker thread, for two reasons: MLX arrays and streams
    belong to the thread that made them, and the event loop must not block on the GPU. One
    thread also serializes access to the model, which is what a single accelerator wants.
    """

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, *, agent: Any = None):
        self.checkpoint = checkpoint
        self.stats = LayaStats()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya")
        self._agent = agent

    def _load(self):
        if self._agent is None:
            try:
                import laya_mlx
            except ImportError as error:  # pragma: no cover - needs the extra
                raise ImportError(
                    "Running Laya needs laya-mlx on Apple silicon: "
                    "pip install 'jev-flywheel[laya]'. Elsewhere, run pambrose/laya-server "
                    "and point TYPESAFE_BASE_URL at it.") from error
            self._agent = laya_mlx.load(self.checkpoint)
        return self._agent

    def _run(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict:
        import time
        agent = self._load()
        state = to_laya_state(state)
        laya_questions = {name: to_laya_question(q) for name, q in questions.items()}
        check_budget(agent, state, laya_questions)
        started = time.perf_counter()
        result = agent.system_one(state, laya_questions)
        self.stats.record(len(laya_questions), time.perf_counter() - started,
                          (result.get("usage") or {}).get("input_tokens") or 0)
        return result

    async def system_one(self, *, state: Any, questions: Mapping[str, Mapping[str, Any]]):
        result = await asyncio.get_running_loop().run_in_executor(
            self._pool, self._run, state, dict(questions))
        return SimpleNamespace(
            answers=result["answers"], usage=result.get("usage"), model=self.model_name)

    @property
    def model_name(self) -> str:
        return f"laya:{self.checkpoint}"

    def warm(self) -> None:
        """Load the checkpoint now, so the first request does not pay for it."""
        self._pool.submit(self._load).result()
