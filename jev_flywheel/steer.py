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


def render_source(*, provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL,
                  max_tokens: int = DEFAULT_MAX_TOKENS, path: Path = PROCEDURE) -> str:
    """The procedure with its provider, model and token budget filled in."""
    source = Path(path).read_text(encoding="utf-8")
    for token, value in (("{{PROVIDER}}", provider), ("{{MODEL}}", model),
                         ("{{MAX_TOKENS}}", str(max_tokens))):
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
) -> SteerOutcome:
    """Run one round of meta-cognition and record it.

    ``mock_replies`` replaces the language model with scripted replies, one per call, so
    the whole loop can be exercised with no API keys. Every other part -- the host, the
    fit, the approvals, the commit -- runs for real.
    """
    return asyncio.run(_run(
        workspace, score_name, provider=provider, model=model, max_tokens=max_tokens,
        allow_spend=allow_spend, client_factory=client_factory, hitl_handler=hitl_handler,
        mock_replies=mock_replies, max_auto_requests=max_auto_requests,
        max_revisions=max_revisions))


async def _run(workspace, score_name, *, provider, model, max_tokens, allow_spend,
               client_factory, hitl_handler, mock_replies, max_auto_requests, max_revisions):
    try:
        from tactus.adapters.memory import MemoryStorage
        from tactus.core.runtime import TactusRuntime
    except ImportError as error:
        raise SteerError(
            "steering needs Tactus. Install it with: pip install 'jev-flywheel[steer]'") from error

    if provider == "bedrock" and not mock_replies:
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    host = FlywheelHost(workspace, score_name, allow_spend=allow_spend,
                        client_factory=client_factory)
    if hitl_handler is None:
        from tactus.adapters.cli_hitl import CLIHITLHandler
        hitl_handler = CLIHITLHandler()

    runtime = TactusRuntime(
        procedure_id=f"steer-{workspace.version}-{len(workspace.feedback())}",
        storage_backend=MemoryStorage(), hitl_handler=hitl_handler,
        source_file_path=str(PROCEDURE))
    runtime.register_python_module("flywheel", host)
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
        runtime.external_agent_mocks = {"analyst": [{"message": r} for r in mock_replies]}

    result = await runtime.execute(
        render_source(provider=provider, model=model, max_tokens=max_tokens),
        context={"max_auto_requests": max_auto_requests, "max_revisions": max_revisions},
        format="lua")
    if not result.get("success"):
        raise SteerError(result.get("error") or "the steering procedure failed")

    detail = _plain(result.get("result")) or {}
    decision = str(detail.get("decision", "unknown"))
    workspace.log_event(
        "rethink", score_name, decision=decision, model=model,
        new_version=detail.get("version"), root_cause=detail.get("root_cause"),
        analyst_reply=host.last_reply)
    return SteerOutcome(
        decision, detail, list(getattr(hitl_handler, "asked", [])))
