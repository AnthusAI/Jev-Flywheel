"""
LLM callable helpers for Reinforcement Memory.

Defines callable type aliases and factory functions for the three LLM tasks:

- **Labeling** (:data:`LabelFn`): Generate a human-readable label for a topic
  cluster from its keywords and representative exemplars.
- **Causal inference** (:data:`CausalFn`): Infer a root cause for a single
  text given optional context (score value, explanation, item text, etc.).
- **Synthesis** (:data:`SynthesisFn`): Synthesize a single root cause
  statement for a topic from per-exemplar causes.

Two labeler/causal/synthesizer implementations are provided:

- ``dspy_*`` — uses Biblicus's existing ``ai/llm.py`` backend.
- ``bedrock_*`` — calls AWS Bedrock directly (Claude Haiku by default),
  compatible with the Plexus usage pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LabelFn = Callable[[List[str], List[str]], str]
"""Callable ``(keywords: list[str], exemplars: list[str]) -> label: str``."""

CausalFn = Callable[[str, Dict[str, Any]], Optional[str]]
"""Callable ``(text: str, context: dict) -> cause: str | None``."""

SynthesisFn = Callable[[str, List[str], List[str]], Optional[str]]
"""Callable ``(label: str, keywords: list[str], causes: list[str]) -> cause: str | None``."""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LABEL_SYSTEM = (
    "You create extremely concise topic labels — 2 to 5 words, title case. "
    "No preamble, no punctuation, no quotes."
)

_LABEL_PROMPT_TEMPLATE = (
    "Keywords: {keywords}\n\n" "Example texts:\n{exemplars}\n\n" "2-5 word topic label:"
)

_CAUSAL_SYSTEM = (
    "You write extremely concise root cause statements — one short sentence, "
    "under 20 words. No preamble, no qualifications."
)

_CAUSAL_PROMPT_TEMPLATE = (
    "Edit comment: {edit_comment}\n" "{context_block}" "One-sentence root cause (under 20 words):"
)

_SYNTHESIS_SYSTEM = (
    "You write extremely concise root cause statements — one short sentence, "
    "under 20 words. No preamble, no qualifications, no restating the cluster name."
)

_SYNTHESIS_PROMPT_TEMPLATE = (
    'Cluster: "{label}" (keywords: {keywords})\n\n'
    "Inferred causes from examples:\n{causes}\n\n"
    "One-sentence root cause (under 20 words):"
)


def _build_context_block(context: Dict[str, Any]) -> str:
    parts = []
    if context.get("score_value"):
        parts.append(f"Score value: {context['score_value']}")
    if context.get("score_explanation"):
        parts.append(f"Explanation: {context['score_explanation'][:500]}")
    if context.get("score_guidelines"):
        parts.append(f"Guidelines: {context['score_guidelines'][:500]}")
    if context.get("item_text_excerpt"):
        parts.append(f"Item text: {context['item_text_excerpt'][:500]}")
    return ("\n".join(parts) + "\n") if parts else ""


# ---------------------------------------------------------------------------
# DSPy-backed helpers
# ---------------------------------------------------------------------------


def dspy_labeler(client=None) -> LabelFn:
    """
    Return a :data:`LabelFn` backed by Biblicus's DSPy LLM backend.

    :param client: Optional :class:`~biblicus.ai.models.LlmClientConfig`.
        Defaults to the environment-configured LLM.
    :return: Label generation callable.
    """
    from biblicus.ai.llm import generate_completion

    def _label(keywords: List[str], exemplars: List[str]) -> str:
        kw_str = ", ".join(keywords[:8])
        ex_str = "\n".join(f"- {e[:200]}" for e in exemplars[:5])
        prompt = _LABEL_PROMPT_TEMPLATE.format(keywords=kw_str, exemplars=ex_str)
        result = generate_completion(
            client=client,
            system_prompt=_LABEL_SYSTEM,
            user_prompt=prompt,
        )
        return result.text.strip() if result and result.text else ", ".join(keywords[:3])

    return _label


def dspy_causal(client=None) -> CausalFn:
    """
    Return a :data:`CausalFn` backed by Biblicus's DSPy LLM backend.

    :param client: Optional :class:`~biblicus.ai.models.LlmClientConfig`.
    :return: Causal inference callable.
    """
    from biblicus.ai.llm import generate_completion

    def _causal(text: str, context: Dict[str, Any]) -> Optional[str]:
        ctx_block = _build_context_block(context)
        prompt = _CAUSAL_PROMPT_TEMPLATE.format(
            edit_comment=text[:500],
            context_block=ctx_block,
        )
        try:
            result = generate_completion(
                client=client,
                system_prompt=_CAUSAL_SYSTEM,
                user_prompt=prompt,
            )
            return result.text.strip() if result and result.text else None
        except Exception as exc:
            logger.warning("dspy_causal LLM call failed: %s", exc)
            return None

    return _causal


def dspy_synthesizer(client=None) -> SynthesisFn:
    """
    Return a :data:`SynthesisFn` backed by Biblicus's DSPy LLM backend.

    :param client: Optional :class:`~biblicus.ai.models.LlmClientConfig`.
    :return: Cause synthesis callable.
    """
    from biblicus.ai.llm import generate_completion

    def _synthesize(label: str, keywords: List[str], causes: List[str]) -> Optional[str]:
        kw_str = ", ".join(keywords[:8])
        causes_str = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(causes))
        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            label=label,
            keywords=kw_str,
            causes=causes_str,
        )
        try:
            result = generate_completion(
                client=client,
                system_prompt=_SYNTHESIS_SYSTEM,
                user_prompt=prompt,
            )
            return result.text.strip() if result and result.text else None
        except Exception as exc:
            logger.warning("dspy_synthesizer LLM call failed: %s", exc)
            return None

    return _synthesize


# ---------------------------------------------------------------------------
# Bedrock-backed helpers
# ---------------------------------------------------------------------------


def _bedrock_call(
    model_id: str,
    region: str,
    system: str,
    prompt: str,
    max_tokens: int = 60,
) -> Optional[str]:
    """Make a single Bedrock Claude invocation and return the text."""
    try:
        import json as _json

        import boto3

        client = boto3.client("bedrock-runtime", region_name=region)
        body = _json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        resp = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = _json.loads(resp["body"].read())
        text = result["content"][0]["text"].strip()
        return text if text else None
    except Exception as exc:
        logger.warning("Bedrock LLM call failed: %s", exc)
        return None


def bedrock_labeler(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
) -> LabelFn:
    """
    Return a :data:`LabelFn` that calls AWS Bedrock Claude.

    :param model_id: Bedrock model ID.
    :param region: AWS region.
    :return: Label generation callable.
    """

    def _label(keywords: List[str], exemplars: List[str]) -> str:
        kw_str = ", ".join(keywords[:8])
        ex_str = "\n".join(f"- {e[:200]}" for e in exemplars[:5])
        prompt = _LABEL_PROMPT_TEMPLATE.format(keywords=kw_str, exemplars=ex_str)
        result = _bedrock_call(model_id, region, _LABEL_SYSTEM, prompt, max_tokens=30)
        return result or ", ".join(keywords[:3])

    return _label


def bedrock_causal(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
) -> CausalFn:
    """
    Return a :data:`CausalFn` that calls AWS Bedrock Claude.

    :param model_id: Bedrock model ID.
    :param region: AWS region.
    :return: Causal inference callable.
    """

    def _causal(text: str, context: Dict[str, Any]) -> Optional[str]:
        ctx_block = _build_context_block(context)
        prompt = _CAUSAL_PROMPT_TEMPLATE.format(
            edit_comment=text[:500],
            context_block=ctx_block,
        )
        return _bedrock_call(model_id, region, _CAUSAL_SYSTEM, prompt, max_tokens=60)

    return _causal


def bedrock_synthesizer(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
) -> SynthesisFn:
    """
    Return a :data:`SynthesisFn` that calls AWS Bedrock Claude.

    :param model_id: Bedrock model ID.
    :param region: AWS region.
    :return: Cause synthesis callable.
    """

    def _synthesize(label: str, keywords: List[str], causes: List[str]) -> Optional[str]:
        kw_str = ", ".join(keywords[:8])
        causes_str = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(causes))
        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            label=label,
            keywords=kw_str,
            causes=causes_str,
        )
        return _bedrock_call(model_id, region, _SYNTHESIS_SYSTEM, prompt, max_tokens=60)

    return _synthesize
