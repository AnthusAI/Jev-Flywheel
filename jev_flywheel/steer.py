"""Running the steering procedure.

Builds the Tactus runtime, registers the Python host as the ``flywheel`` module the
procedure requires, hands it a human-approval channel, and runs one round. Everything
here is plumbing; the rules live in ``procedures/steer_scorecard.tac`` (the loop) and
``host.py`` (the numbers).

Tactus is an optional dependency (``pip install jev-flywheel[steer]``), so it is imported
only when a round actually runs.

Providers: Tactus supports ``bedrock`` and ``openai``. The default is Kimi K3 on Bedrock,
through the ``us.`` inference profile. K3 is invoked by profile rather than by bare model
id, spends hidden reasoning tokens (hence the generous ``max_tokens``), and does not follow
Tactus's structured-output protocol, which is why the agent replies in JSON text and the
host parses it. If you use ``aws login`` credentials, boto needs ``botocore[crt]``.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from jev_flywheel.host import FlywheelHost, _plain
from jev_flywheel.workspace import Workspace

DEFAULT_PROVIDER = "bedrock"
DEFAULT_MODEL = "us.moonshotai.kimi-k3"
DEFAULT_MAX_TOKENS = 8000
PROCEDURE = Path(__file__).resolve().parents[1] / "procedures" / "steer_scorecard.tac"


class SteerError(RuntimeError):
    """The steering round could not run. The message says why."""


def apply_region(provider: str, region: Optional[str]) -> None:
    """Point Bedrock at a region. Models are not offered everywhere: Qwen3 Coder 480B is in
    us-west-2 and us-east-2 but not us-east-1, so the region is a per-model choice.

    An explicit region wins; otherwise an already-configured one is kept; otherwise us-east-1.
    """
    if provider != "bedrock":
        return
    if region:
        os.environ["AWS_DEFAULT_REGION"] = region
    else:
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


GATE_NOTE_TEMPLATE = (
    "\nOne more rule for this round: a promotion gate will reject any new element whose own "
    "answers change on more than {rate:.0%} of the labeled items when the item's gender is "
    "swapped (pronouns and a short list of role nouns). You will not be told which items or "
    "what the swap changed -- write each element so it could be answered the same way "
    "regardless of the subject's gender, the same discipline good survey questions already "
    "follow.\n"
)


def render_source(*, provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL,
                  max_tokens: int = DEFAULT_MAX_TOKENS, path: Path = PROCEDURE,
                  invariance_max_flip_rate: Optional[float] = None) -> str:
    """The procedure with its provider, model, token budget and (optionally) the invariance
    gate's existence filled in.

    ``invariance_max_flip_rate`` is ``studies/PREREGISTERED.md``'s J2/L2 arms: when given, the
    analyst is told a gate exists and what it measures (never how it is computed, and never
    which items it runs on), so a rejected proposal cannot be traced back to specific items.
    """
    source = Path(path).read_text(encoding="utf-8")
    gate_note = (GATE_NOTE_TEMPLATE.format(rate=invariance_max_flip_rate)
                if invariance_max_flip_rate is not None else "")
    for token, value in (("{{PROVIDER}}", provider), ("{{MODEL}}", model),
                         ("{{MAX_TOKENS}}", str(max_tokens)), ("{{GATE_NOTE}}", gate_note)):
        source = source.replace(token, value)
    return source


class ScriptedApprover:
    """A human-approval channel that answers from a script, and remembers what it was asked.

    For tests and for replaying a recorded run. It is not a way to skip review: the
    answers are the recorded decisions of a person, and each question is kept so a
    reader can see exactly what they were shown.
    """

    def __init__(self, answers: Sequence[bool] = (), default: bool = False):
        self._answers = list(answers)
        self.default = default
        self.asked: List[str] = []

    def request_interaction(self, procedure_id, request, execution_context=None):
        from tactus.protocols.models import HITLResponse

        self.asked.append(request.message)
        value = self._answers.pop(0) if self._answers else self.default
        return HITLResponse(value=value, responded_at=datetime.now())

    def check_pending_response(self, procedure_id, message_id):
        return None

    def cancel_pending_request(self, procedure_id, message_id):
        return None


@dataclass
class SteerOutcome:
    """What a round decided, and everything it was shown and asked."""

    decision: str
    detail: Dict[str, Any] = field(default_factory=dict)
    approvals_asked: List[str] = field(default_factory=list)
    analyst_reply: Optional[str] = None
    discovery_reply: Optional[str] = None

    @property
    def promoted(self) -> bool:
        return self.decision == "promoted"


def run_steering(
    workspace: Workspace,
    score_name: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    allow_spend: bool = False,
    client_factory: Optional[Callable[[], Any]] = None,
    hitl_handler: Any = None,
    mock_replies: Optional[Sequence[str]] = None,
    max_auto_requests: int = 300,
    max_revisions: int = 1,
    region: Optional[str] = None,
    max_mismatches: int = 25,
    discovery: bool = False,
    taxonomy: bool = False,
    invariance_max_flip_rate: Optional[float] = None,
) -> SteerOutcome:
    """Run one round of meta-cognition and record it.

    ``mock_replies`` replaces the language model with scripted replies, one per call, so
    the whole loop can be exercised with no API keys. Every other part -- the host, the
    fit, the approvals, the commit -- runs for real.

    ``invariance_max_flip_rate`` turns on the gender-invariance promotion gate
    (``jev_flywheel.invariance``, ``studies/PREREGISTERED.md``'s J2/L2 arms): ``None`` (the
    default) reproduces every existing caller's behaviour exactly. When given, a proposed
    element is rejected -- on top of, never instead of, the ordinary out-of-fold fit test -- if
    its own answers flip on more than this share of the labeled items under
    ``jev_flywheel.counterfactual.swap_gender``, and the analyst's briefing says the gate exists.
    """
    return asyncio.run(_run(
        workspace, score_name, provider=provider, model=model, max_tokens=max_tokens,
        allow_spend=allow_spend, client_factory=client_factory, hitl_handler=hitl_handler,
        mock_replies=mock_replies, max_auto_requests=max_auto_requests,
        max_revisions=max_revisions, region=region, max_mismatches=max_mismatches,
        discovery=discovery, taxonomy=taxonomy,
        invariance_max_flip_rate=invariance_max_flip_rate))


async def _run(workspace, score_name, *, provider, model, max_tokens, allow_spend,
               client_factory, hitl_handler, mock_replies, max_auto_requests, max_revisions,
               region=None, max_mismatches=25, discovery=False, taxonomy=False,
               invariance_max_flip_rate=None):
    try:
        from tactus.adapters.memory import MemoryStorage
        from tactus.core.runtime import TactusRuntime
    except ImportError as error:
        raise SteerError(
            "steering needs Tactus. Install it with: pip install 'jev-flywheel[steer]'") from error

    if not mock_replies:
        apply_region(provider, region)
    host = FlywheelHost(workspace, score_name, allow_spend=allow_spend,
                        client_factory=client_factory, max_mismatches=max_mismatches,
                        invariance_max_flip_rate=invariance_max_flip_rate)
    if hitl_handler is None:
        from tactus.adapters.cli_hitl import CLIHITLHandler
        hitl_handler = CLIHITLHandler()

    runtime = TactusRuntime(
        procedure_id=f"steer-{workspace.version}-{len(workspace.feedback())}",
        storage_backend=MemoryStorage(), hitl_handler=hitl_handler,
        source_file_path=str(PROCEDURE))
    runtime.register_python_module("flywheel", host)
    if mock_replies is not None and discovery:
        # The blind pass consumes a turn too, so a scripted round needs one reply per call.
        mock_replies = list(mock_replies)
        if len(mock_replies) == 1:
            mock_replies = mock_replies * 2
    if mock_replies is not None:
        # Agents are created while the procedure is parsed, so mocking must be switched on
        # before that, and it takes a MockManager as well as the scripted turns. Without
        # the manager the agent is built against the real model, which is a live (and paid)
        # call: an easy mistake to make, so it is done in one place and spec'd.
        from tactus.core.mocking import MockManager, set_current_mock_manager

        manager = MockManager()
        runtime.mock_manager = manager
        set_current_mock_manager(manager)
        runtime.mock_all_agents = True
        turns = [{"message": r} for r in mock_replies]
        runtime.external_agent_mocks = {"analyst": turns, "scout": turns}

    result = await runtime.execute(
        render_source(provider=provider, model=model, max_tokens=max_tokens,
                     invariance_max_flip_rate=invariance_max_flip_rate),
        context={"max_auto_requests": max_auto_requests, "max_revisions": max_revisions,
                 "discovery": discovery, "taxonomy": taxonomy},
        format="lua")
    if not result.get("success"):
        raise SteerError(result.get("error") or "the steering procedure failed")

    detail = _plain(result.get("result")) or {}
    decision = str(detail.get("decision", "unknown"))
    workspace.log_event(
        "rethink", score_name, decision=decision, model=model,
        new_version=detail.get("version"), root_cause=detail.get("root_cause"),
        status=detail.get("status"), reason=detail.get("reason"),
        analyst_reply=host.last_reply)
    return SteerOutcome(
        decision, detail, list(getattr(hitl_handler, "asked", [])), host.last_reply,
        host.last_discovery_reply)
