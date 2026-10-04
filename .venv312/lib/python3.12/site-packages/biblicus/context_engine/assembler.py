"""
Context assembly utilities for the Biblicus Context Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, Iterable, Optional

from biblicus.context import ContextPack, ContextPackBlock
from biblicus.context_engine.compaction import CompactionRequest, TruncateCompactor, build_compactor
from biblicus.context_engine.models import (
    AssistantMessageSpec,
    ContextDeclaration,
    ContextInsertSpec,
    ContextMessageSpec,
    ContextPolicySpec,
    ContextRetrieverRequest,
    HistoryInsertSpec,
    SystemMessageSpec,
    UserMessageSpec,
)


@dataclass
class ContextAssemblyResult:
    """
    Assembled prompt context for a single agent turn.

    :ivar system_prompt: System prompt content.
    :vartype system_prompt: str
    :ivar history: Message history for the turn.
    :vartype history: list[dict[str, Any]]
    :ivar user_message: User message content.
    :vartype user_message: str
    :ivar token_count: Estimated token count for assembled content.
    :vartype token_count: int
    """

    system_prompt: str
    history: list[dict[str, Any]]
    user_message: str
    token_count: int = 0


class ContextAssembler:
    """
    Assemble Context declarations into system prompts, history, and user messages.

    :param context_registry: Context declarations indexed by name.
    :type context_registry: dict[str, ContextDeclaration]
    :param retriever_registry: Retriever declarations indexed by name.
    :type retriever_registry: dict[str, Any] or None
    :param corpus_registry: Corpus declarations indexed by name.
    :type corpus_registry: dict[str, Any] or None
    :param compactor_registry: Compactor declarations indexed by name.
    :type compactor_registry: dict[str, Any] or None
    :param default_retriever: Default retriever callable when no override is supplied.
    :type default_retriever: callable or None
    """

    def __init__(
        self,
        context_registry: dict[str, ContextDeclaration],
        retriever_registry: Optional[dict[str, Any]] = None,
        corpus_registry: Optional[dict[str, Any]] = None,
        compactor_registry: Optional[dict[str, Any]] = None,
        default_retriever: Optional[Any] = None,
    ):
        self._context_registry = context_registry
        self._retriever_registry = retriever_registry or {}
        self._corpus_registry = corpus_registry or {}
        self._compactor_registry = compactor_registry or {}
        self._default_retriever = default_retriever

    def assemble(
        self,
        context_name: str,
        base_system_prompt: str,
        history_messages: list[dict[str, Any]],
        user_message: Optional[str],
        template_context: dict[str, Any],
        retriever_override: Optional[Any] = None,
    ) -> ContextAssemblyResult:
        """
        Assemble a Context declaration into prompt components.

        :param context_name: Name of the Context declaration.
        :type context_name: str
        :param base_system_prompt: Default system prompt from agent config.
        :type base_system_prompt: str
        :param history_messages: Current agent history messages.
        :type history_messages: list[dict[str, Any]]
        :param user_message: Current user message for this turn.
        :type user_message: str or None
        :param template_context: Template variables for resolution.
        :type template_context: dict[str, Any]
        :param retriever_override: Optional retriever override callable.
        :type retriever_override: callable or None
        :return: Assembled prompt components.
        :rtype: ContextAssemblyResult
        :raises ValueError: If the context declaration is not found.
        """
        if context_name not in self._context_registry:
            raise ValueError(f"Context '{context_name}' not defined")

        context_spec = self._context_registry[context_name]
        if context_spec.messages is None:
            return self._assemble_default_with_regeneration(
                context_spec,
                base_system_prompt,
                history_messages,
                user_message or "",
                template_context,
                retriever_override,
            )

        return self._assemble_explicit_with_regeneration(
            context_spec,
            history_messages,
            user_message or "",
            template_context,
            retriever_override,
        )

    def _assemble_default(
        self,
        context_spec: ContextDeclaration,
        base_system_prompt: str,
        history_messages: list[dict[str, Any]],
        user_message: str,
        template_context: dict[str, Any],
        total_budget_override: Optional[int] = None,
    ) -> ContextAssemblyResult:
        """Assemble the default Context plan when messages are omitted."""
        system_prompt = base_system_prompt or ""
        pack_outputs = []
        pack_entries = context_spec.packs or []
        pack_budgets = self._allocate_default_pack_budgets(
            pack_entries, context_spec.policy, total_budget_override
        )
        for pack_entry in pack_entries:
            pack_outputs.append(
                self._render_pack(
                    pack_entry.name,
                    template_context,
                    retriever_override=None,
                    pack_budget=pack_budgets.get(pack_entry.name),
                    policy=context_spec.policy,
                    weight=pack_entry.weight,
                )
            )

        if pack_outputs:
            system_prompt = self._join_nonempty([system_prompt, *pack_outputs])

        compacted_prompt, history_messages, user_message, token_count, compacted = (
            self._apply_context_budget(
                system_prompt,
                history_messages,
                user_message,
                context_spec.policy,
            )
        )
        return ContextAssemblyResult(
            system_prompt=compacted_prompt,
            history=history_messages,
            user_message=user_message,
            token_count=token_count,
        )

    def _assemble_default_with_regeneration(
        self,
        context_spec: ContextDeclaration,
        base_system_prompt: str,
        history_messages: list[dict[str, Any]],
        user_message: str,
        template_context: dict[str, Any],
        retriever_override: Optional[Any],
        total_budget_override: Optional[int] = None,
    ) -> ContextAssemblyResult:
        max_iterations = 2
        if context_spec.policy and getattr(context_spec.policy, "max_iterations", None):
            max_iterations = max(1, int(context_spec.policy.max_iterations))

        pack_scale = 1.0
        last_result: Optional[ContextAssemblyResult] = None
        for _iteration in range(max_iterations):
            system_prompt = base_system_prompt or ""
            pack_outputs = []
            pack_entries = context_spec.packs or []
            pack_budgets = self._allocate_default_pack_budgets(
                pack_entries, context_spec.policy, total_budget_override
            )
            for pack_entry in pack_entries:
                pack_outputs.append(
                    self._render_pack(
                        pack_entry.name,
                        template_context,
                        retriever_override,
                        pack_budget=pack_budgets.get(pack_entry.name),
                        policy=context_spec.policy,
                        tighten_pack_budget=pack_scale < 1.0,
                        weight=pack_entry.weight,
                    )
                )

            if pack_outputs:
                system_prompt = self._join_nonempty([system_prompt, *pack_outputs])

            compacted_prompt, history_messages, user_message, token_count, compacted = (
                self._apply_context_budget(
                    system_prompt,
                    history_messages,
                    user_message,
                    context_spec.policy,
                )
            )

            last_result = ContextAssemblyResult(
                system_prompt=compacted_prompt,
                history=history_messages,
                user_message=user_message,
                token_count=token_count,
            )

            if not compacted or not context_spec.policy:
                break
            if getattr(context_spec.policy, "overflow", None) != "compact":
                break
            pack_scale *= 0.5

        return last_result or ContextAssemblyResult(
            system_prompt="",
            history=history_messages,
            user_message=user_message,
            token_count=0,
        )

    def _assemble_explicit_with_regeneration(
        self,
        context_spec: ContextDeclaration,
        history_messages: list[dict[str, Any]],
        user_message: str,
        template_context: dict[str, Any],
        retriever_override: Optional[Any],
    ) -> ContextAssemblyResult:
        """Assemble explicit Context message directives with regeneration loop."""
        max_iterations = 2
        if context_spec.policy and getattr(context_spec.policy, "max_iterations", None):
            max_iterations = max(1, int(context_spec.policy.max_iterations))

        pack_scale = 1.0
        last_result: Optional[ContextAssemblyResult] = None
        for _iteration in range(max_iterations):
            total_pack_budget = self._resolve_default_pack_total_budget(context_spec.policy)
            if total_pack_budget is not None and pack_scale < 1.0:
                total_pack_budget = max(1, int(total_pack_budget * pack_scale))

            assembled_messages = self._build_messages(
                context_spec,
                history_messages,
                template_context,
                retriever_override,
                context_spec.policy,
                tighten_pack_budget=pack_scale < 1.0,
                total_pack_budget_override=total_pack_budget,
            )

            system_messages, remaining_messages = self._split_leading_system(assembled_messages)
            system_prompt = self._join_nonempty([m["content"] for m in system_messages])
            resolved_user_message, remaining_messages = self._extract_user_message(
                remaining_messages, user_message
            )

            compacted_prompt, remaining_messages, resolved_user_message, token_count, compacted = (
                self._apply_context_budget(
                    system_prompt,
                    remaining_messages,
                    resolved_user_message,
                    context_spec.policy,
                )
            )

            last_result = ContextAssemblyResult(
                system_prompt=compacted_prompt,
                history=remaining_messages,
                user_message=resolved_user_message,
                token_count=token_count,
            )

            if not compacted or not context_spec.policy:
                break
            if getattr(context_spec.policy, "overflow", None) != "compact":
                break
            pack_scale *= 0.5

        return last_result or ContextAssemblyResult(
            system_prompt="",
            history=[],
            user_message=user_message,
            token_count=0,
        )

    def _resolve_message_content(
        self,
        directive: SystemMessageSpec | UserMessageSpec | AssistantMessageSpec,
        template_context,
    ) -> str:
        """Resolve message content or templates."""
        if directive.content is not None:
            return directive.content
        return self._resolve_template(directive.template or "", directive.vars, template_context)

    def _resolve_template(
        self, template_text: str, vars_dict: dict[str, Any], template_context: dict[str, Any]
    ) -> str:
        """Resolve dot-notation templates with context variables."""
        if not template_text:
            return template_text

        class DotFormatter(Formatter):
            def get_field(self, field_name, args, kwargs):
                path_parts = field_name.split(".")
                current_value = kwargs
                for part in path_parts:
                    if isinstance(current_value, dict):
                        current_value = current_value.get(part, "")
                    else:
                        current_value = getattr(current_value, part, "")
                return current_value, field_name

        merged_context = dict(template_context)
        for key, value in (vars_dict or {}).items():
            merged_context[key] = value

        formatter = DotFormatter()
        return formatter.format(template_text, **merged_context)

    def _render_pack(
        self,
        pack_name: str,
        template_context: dict[str, Any],
        retriever_override: Optional[Any],
        pack_budget: Optional[Any],
        policy: Optional[ContextPolicySpec],
        tighten_pack_budget: bool = False,
        weight: Optional[float] = None,
    ) -> str:
        """Render a context pack by name."""
        if pack_name in self._context_registry:
            nested_context = self._context_registry[pack_name]
            return self._render_nested_context_pack(
                nested_context,
                template_context,
                pack_budget,
                policy,
                tighten_pack_budget,
                retriever_override,
            )
        if pack_name in self._retriever_registry:
            return self._render_retriever_pack(
                pack_name,
                template_context,
                retriever_override,
                pack_budget,
                policy,
                tighten_pack_budget,
                weight,
            )
        raise NotImplementedError(
            f"Context pack '{pack_name}' is not available. Only Context or retriever packs are supported."
        )

    def _render_retriever_pack(
        self,
        retriever_name: str,
        template_context: dict[str, Any],
        retriever_override: Optional[Any],
        pack_budget: Optional[Any],
        policy: Optional[ContextPolicySpec],
        tighten_pack_budget: bool,
        weight: Optional[float] = None,
    ) -> str:
        """Render a retriever pack for the given retriever."""
        retriever_spec = self._retriever_registry[retriever_name]
        config = retriever_spec.config if hasattr(retriever_spec, "config") else {}
        query_template = config.get("query") if isinstance(config, dict) else None
        query = ""
        if isinstance(query_template, str):
            query = self._resolve_template(query_template, {}, template_context)
        if not query:
            input_context = template_context.get("input", {})
            query = input_context.get("query", "") or input_context.get("message", "")

        split = "train"
        maximum_cache_total_items = None
        maximum_cache_total_characters = None
        limit = 3
        offset = 0
        maximum_total_characters = None
        maximum_items_per_source = None
        include_metadata = False
        metadata_fields = None
        retriever_id = None
        corpus_root = None
        snapshot_id = None
        configuration_name = None
        configuration = None
        corpus_name = getattr(retriever_spec, "corpus", None)
        join_with = "\n\n"
        pipeline_config = None
        query_config = None

        if isinstance(config, dict):
            split = config.get("split", split)
            limit = config.get("limit", limit)
            offset = config.get("offset", offset)
            maximum_total_characters = config.get(
                "maximum_total_characters", maximum_total_characters
            )
            maximum_items_per_source = config.get(
                "maximum_items_per_source",
                config.get("max_items_per_source", maximum_items_per_source),
            )
            include_metadata = config.get("include_metadata", include_metadata)
            metadata_fields = config.get("metadata_fields", metadata_fields)
            retriever_id = config.get("retriever_id", retriever_id)
            snapshot_id = config.get("snapshot_id", snapshot_id)
            configuration_name = config.get("configuration_name", configuration_name)
            if isinstance(config.get("configuration"), dict):
                configuration = config.get("configuration")
            corpus_name = config.get("corpus", corpus_name)
            join_with = config.get("join_with", join_with)
            if isinstance(configuration, dict):
                pipeline_config = configuration.get("pipeline")
            if not isinstance(pipeline_config, dict) and isinstance(config.get("pipeline"), dict):
                pipeline_config = config.get("pipeline")
            if isinstance(pipeline_config, dict):
                if isinstance(pipeline_config.get("query"), dict):
                    query_config = pipeline_config.get("query") or {}
                if configuration is None and isinstance(pipeline_config.get("index"), dict):
                    configuration = pipeline_config.get("index") or {}
            if configuration is None and isinstance(config.get("index"), dict):
                configuration = config.get("index") or {}
            if configuration is None and isinstance(pipeline_config, dict):
                configuration = pipeline_config.get("index") or {}
        if corpus_name and corpus_name in self._corpus_registry:
            corpus_spec = self._corpus_registry[corpus_name]
            corpus_config = corpus_spec.config if hasattr(corpus_spec, "config") else {}
            if isinstance(corpus_config, dict):
                split = corpus_config.get("split", split)
                maximum_cache_total_items = corpus_config.get(
                    "maximum_cache_total_items", maximum_cache_total_items
                )
                maximum_cache_total_characters = corpus_config.get(
                    "maximum_cache_total_characters", maximum_cache_total_characters
                )
                corpus_root = corpus_config.get(
                    "corpus_root",
                    corpus_config.get("root", corpus_root),
                )
        if query_config:
            if "limit" in query_config:
                limit = query_config.get("limit", limit)
            if "offset" in query_config:
                offset = query_config.get("offset", offset)
            if "maximum_total_characters" in query_config:
                maximum_total_characters = query_config.get(
                    "maximum_total_characters", maximum_total_characters
                )
            if "maximum_items_per_source" in query_config:
                maximum_items_per_source = query_config.get(
                    "maximum_items_per_source",
                    maximum_items_per_source,
                )
            if "max_items_per_source" in query_config and maximum_items_per_source is None:
                maximum_items_per_source = query_config.get("max_items_per_source")
            if "include_metadata" in query_config:
                include_metadata = query_config.get("include_metadata", include_metadata)
            if "metadata_fields" in query_config:
                metadata_fields = query_config.get("metadata_fields", metadata_fields)
            if "join_with" in query_config:
                join_with = query_config.get("join_with", join_with)

        allocated_tokens = self._allocate_pack_budget(pack_budget, policy, weight)
        if allocated_tokens is not None:
            derived_chars = int(allocated_tokens) * 4
            if maximum_total_characters is None:
                maximum_total_characters = derived_chars
            else:
                maximum_total_characters = min(maximum_total_characters, derived_chars)

        if tighten_pack_budget:
            if maximum_total_characters is not None:
                maximum_total_characters = max(1, int(maximum_total_characters * 0.5))
            limit = max(1, int(limit * 0.5))

        retriever_fn = retriever_override or self._default_retriever
        if retriever_fn is None:
            raise ValueError("No retriever override or default retriever configured")

        request = ContextRetrieverRequest(
            query=query,
            offset=offset,
            limit=limit,
            maximum_total_characters=maximum_total_characters,
            max_tokens=allocated_tokens,
            metadata={
                "retriever": retriever_name,
                "corpus": corpus_name,
                "split": split,
                "maximum_cache_total_items": maximum_cache_total_items,
                "maximum_cache_total_characters": maximum_cache_total_characters,
                "maximum_items_per_source": maximum_items_per_source,
                "include_metadata": include_metadata,
                "metadata_fields": metadata_fields,
                "retriever_id": retriever_id,
                "corpus_root": corpus_root,
                "snapshot_id": snapshot_id,
                "configuration_name": configuration_name,
                "configuration": configuration,
            },
        )
        context_pack = self._retrieve_with_expansion(
            retriever_fn,
            request,
            policy,
            join_with,
            allocated_tokens,
        )
        return context_pack.text

    def _retrieve_with_expansion(
        self,
        retriever_fn: Any,
        request: ContextRetrieverRequest,
        policy: Optional[ContextPolicySpec],
        join_with: str,
        target_tokens: Optional[int],
    ) -> ContextPack:
        packs: list[ContextPack] = []
        expansion = policy.expansion if policy else None
        max_pages = 1
        min_fill_ratio = None
        if expansion is not None:
            max_pages = max(1, expansion.max_pages)
            min_fill_ratio = expansion.min_fill_ratio

        current_request = request
        for page_index in range(max_pages):
            response_pack = retriever_fn(current_request)
            if response_pack is None:
                break
            packs.append(response_pack)

            if max_pages <= 1 or target_tokens is None:
                break
            if response_pack.evidence_count < current_request.limit:
                break

            merged_pack = self._merge_context_packs(packs, join_with=join_with)
            token_count = self._estimate_tokens(merged_pack.text)
            threshold_ratio = 1.0 if min_fill_ratio is None else float(min_fill_ratio)
            if token_count >= int(target_tokens * threshold_ratio):
                break

            current_request = current_request.model_copy(
                update={"offset": current_request.offset + current_request.limit}
            )

        return self._merge_context_packs(packs, join_with=join_with)

    def _merge_context_packs(self, packs: Iterable[ContextPack], join_with: str) -> ContextPack:
        blocks: list[ContextPackBlock] = []
        for index, pack in enumerate(packs, start=1):
            if pack.blocks:
                blocks.extend(pack.blocks)
                continue
            if pack.text:
                blocks.append(
                    ContextPackBlock(
                        evidence_item_id=f"page-{index}",
                        text=pack.text,
                        metadata=None,
                    )
                )
        if not blocks:
            return ContextPack(text="", evidence_count=0, blocks=[])
        text = join_with.join([block.text for block in blocks])
        return ContextPack(text=text, evidence_count=len(blocks), blocks=blocks)

    def _apply_compaction_policy(
        self, text: str, policy: Any, max_tokens_override: Optional[int] = None
    ) -> str:
        if not policy or not getattr(policy, "input_budget", None):
            return text
        max_tokens = max_tokens_override
        if max_tokens is None:
            budget = policy.input_budget
            max_tokens = getattr(budget, "max_tokens", None)
        if max_tokens is None:
            return text
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        if getattr(policy, "overflow", None) != "compact":
            return text

        compactor = self._resolve_compactor(policy)
        return compactor.compact(CompactionRequest(text=text, max_tokens=max_tokens))

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split())

    def _apply_context_budget(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_message: str,
        policy: Any,
    ) -> tuple[str, list[dict[str, Any]], str, int, bool]:
        if not policy or not getattr(policy, "input_budget", None):
            token_count = self._estimate_total_tokens(system_prompt, history, user_message)
            return system_prompt, history, user_message, token_count, False

        budget = policy.input_budget
        max_tokens = getattr(budget, "max_tokens", None)
        if max_tokens is None:
            token_count = self._estimate_total_tokens(system_prompt, history, user_message)
            return system_prompt, history, user_message, token_count, False

        token_count = self._estimate_total_tokens(system_prompt, history, user_message)
        if token_count <= max_tokens:
            return system_prompt, history, user_message, token_count, False

        if getattr(policy, "overflow", None) != "compact":
            return system_prompt, history, user_message, token_count, False

        trimmed_history = list(history)
        compacted = False
        while trimmed_history and token_count > max_tokens:
            trimmed_history.pop(0)
            token_count = self._estimate_total_tokens(system_prompt, trimmed_history, user_message)
            compacted = True

        if token_count <= max_tokens:
            return system_prompt, trimmed_history, user_message, token_count, compacted

        remaining_budget = max_tokens - self._estimate_total_tokens(
            "", trimmed_history, user_message
        )
        if remaining_budget < 0:
            remaining_budget = 0
        compacted_prompt = self._apply_compaction_policy(
            system_prompt,
            policy,
            max_tokens_override=remaining_budget,
        )
        if compacted_prompt != system_prompt:
            compacted = True
        token_count = self._estimate_total_tokens(compacted_prompt, trimmed_history, user_message)
        return compacted_prompt, trimmed_history, user_message, token_count, compacted

    def _allocate_pack_budget(
        self,
        pack_budget: Optional[Any],
        policy: Optional[ContextPolicySpec],
        weight: Optional[float],
    ) -> Optional[int]:
        if pack_budget is not None and hasattr(pack_budget, "max_tokens"):
            max_tokens = getattr(pack_budget, "max_tokens", None)
            if max_tokens is not None:
                return int(max_tokens)
            ratio = getattr(pack_budget, "ratio", None)
            if ratio is not None:
                return self._resolve_ratio_budget(ratio, policy)
        if pack_budget and isinstance(pack_budget, dict):
            max_tokens = pack_budget.get("max_tokens")
            if max_tokens is not None:
                return int(max_tokens)
            ratio = pack_budget.get("ratio")
            if ratio is not None:
                return self._resolve_ratio_budget(ratio, policy)

        if policy and getattr(policy, "pack_budget", None):
            pack_budget_spec = policy.pack_budget
            base_tokens = getattr(pack_budget_spec, "default_max_tokens", None)
            if base_tokens is None:
                base_ratio = getattr(pack_budget_spec, "default_ratio", None)
                if base_ratio is None:
                    return None
                if not getattr(policy, "input_budget", None):
                    return None
                input_budget = policy.input_budget
                max_tokens = getattr(input_budget, "max_tokens", None)
                if max_tokens is None:
                    return None
                base_tokens = int(max_tokens * base_ratio)
            if weight is None:
                return int(base_tokens)
            return int(base_tokens * weight)

        return None

    def _resolve_ratio_budget(
        self, ratio: float, policy: Optional[ContextPolicySpec]
    ) -> Optional[int]:
        if ratio is None or not policy or not getattr(policy, "input_budget", None):
            return None
        input_budget = policy.input_budget
        max_tokens = getattr(input_budget, "max_tokens", None)
        if max_tokens is None:
            return None
        return int(max_tokens * float(ratio))

    def _allocate_directive_budgets(
        self,
        directives: list[ContextMessageSpec],
        policy: Optional[ContextPolicySpec],
        total_budget_override: Optional[int],
    ) -> dict[int, dict[str, Any]]:
        if not policy or not getattr(policy, "pack_budget", None):
            return {}
        pack_budget_spec = policy.pack_budget
        total_budget = total_budget_override
        if total_budget is None:
            total_budget = getattr(pack_budget_spec, "default_max_tokens", None)
        if total_budget is None:
            base_ratio = getattr(pack_budget_spec, "default_ratio", None)
            if base_ratio is None:
                return {}
            if not getattr(policy, "input_budget", None):
                return {}
            input_budget = policy.input_budget
            max_tokens = getattr(input_budget, "max_tokens", None)
            if max_tokens is None:
                return {}
            total_budget = int(max_tokens * base_ratio)

        pack_directives = [
            directive for directive in directives if isinstance(directive, ContextInsertSpec)
        ]
        if not pack_directives:
            return {}

        sorted_directives = sorted(
            pack_directives,
            key=lambda directive: (
                -(directive.priority or 0),
                -(directive.weight or 1.0),
            ),
        )
        total_weight = sum(directive.weight or 1.0 for directive in sorted_directives)
        allocations = {}
        remaining_budget = int(total_budget)

        for directive in sorted_directives:
            weight = directive.weight or 1.0
            allocation = int((total_budget * weight) / total_weight)
            if allocation <= 0:
                allocation = 1
            if allocation > remaining_budget:
                allocation = remaining_budget
            allocations[id(directive)] = {"max_tokens": allocation}
            remaining_budget -= allocation
            if remaining_budget <= 0:
                break

        if remaining_budget > 0 and sorted_directives:
            allocations[id(sorted_directives[0])]["max_tokens"] += remaining_budget

        return allocations

    def _allocate_default_pack_budgets(
        self,
        pack_entries: list[Any],
        policy: Optional[ContextPolicySpec],
        total_budget_override: Optional[int],
    ) -> dict[str, dict[str, Any]]:
        if not policy or not getattr(policy, "pack_budget", None):
            return {}
        pack_budget_spec = policy.pack_budget
        total_budget = total_budget_override
        if total_budget is None:
            total_budget = getattr(pack_budget_spec, "default_max_tokens", None)
        if total_budget is None:
            base_ratio = getattr(pack_budget_spec, "default_ratio", None)
            if base_ratio is None:
                return {}
            if not getattr(policy, "input_budget", None):
                return {}
            input_budget = policy.input_budget
            max_tokens = getattr(input_budget, "max_tokens", None)
            if max_tokens is None:
                return {}
            total_budget = int(max_tokens * base_ratio)

        if not pack_entries:
            return {}

        entries = sorted(
            pack_entries,
            key=lambda entry: (
                -(getattr(entry, "priority", None) or 0),
                -(getattr(entry, "weight", None) or 1.0),
            ),
        )
        total_weight = sum(getattr(entry, "weight", None) or 1.0 for entry in entries)
        allocations = {}
        remaining_budget = int(total_budget)

        for entry in entries:
            weight = getattr(entry, "weight", None) or 1.0
            allocation = int((total_budget * weight) / total_weight)
            if allocation <= 0:
                allocation = 1
            if allocation > remaining_budget:
                allocation = remaining_budget
            allocations[entry.name] = {"max_tokens": allocation}
            remaining_budget -= allocation
            if remaining_budget <= 0:
                break

        if remaining_budget > 0 and entries:
            allocations[entries[0].name]["max_tokens"] += remaining_budget

        return allocations

    def _resolve_default_pack_total_budget(
        self, policy: Optional[ContextPolicySpec]
    ) -> Optional[int]:
        if not policy or not getattr(policy, "pack_budget", None):
            return None
        pack_budget_spec = policy.pack_budget
        total_budget = getattr(pack_budget_spec, "default_max_tokens", None)
        if total_budget is not None:
            return int(total_budget)
        base_ratio = getattr(pack_budget_spec, "default_ratio", None)
        if base_ratio is None:
            return None
        if not getattr(policy, "input_budget", None):
            return None
        input_budget = policy.input_budget
        max_tokens = getattr(input_budget, "max_tokens", None)
        if max_tokens is None:
            return None
        return int(max_tokens * base_ratio)

    def _extract_pack_budget_max_tokens(
        self, pack_budget: Optional[Any], policy: Optional[ContextPolicySpec]
    ) -> Optional[int]:
        if pack_budget is None:
            return None
        if isinstance(pack_budget, dict):
            max_tokens = pack_budget.get("max_tokens")
            if max_tokens is not None:
                return int(max_tokens)
            ratio = pack_budget.get("ratio")
            if ratio is not None:
                return self._resolve_ratio_budget(ratio, policy)
        if hasattr(pack_budget, "max_tokens"):
            max_tokens = getattr(pack_budget, "max_tokens", None)
            if max_tokens is not None:
                return int(max_tokens)
            ratio = getattr(pack_budget, "ratio", None)
            if ratio is not None:
                return self._resolve_ratio_budget(ratio, policy)
        return None

    def _estimate_total_tokens(
        self, system_prompt: str, history: list[dict[str, Any]], user_message: str
    ) -> int:
        total = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_message)
        for message in history:
            content = message.get("content", "")
            total += self._estimate_tokens(content)
        return total

    def _resolve_compactor(self, policy: ContextPolicySpec):
        compactor_config = None
        if hasattr(policy, "compactor"):
            compactor_config = policy.compactor

        if isinstance(compactor_config, str):
            compactor_spec = self._compactor_registry.get(compactor_config)
            if compactor_spec is None:
                raise ValueError(f"Compactor '{compactor_config}' not defined")
            config = compactor_spec.config if hasattr(compactor_spec, "config") else {}
            if not isinstance(config, dict):
                config = {}
            return build_compactor(config)

        if isinstance(compactor_config, dict):
            return build_compactor(compactor_config)

        return TruncateCompactor()

    def _build_messages(
        self,
        context_spec: ContextDeclaration,
        history_messages: list[dict[str, Any]],
        template_context: dict[str, Any],
        retriever_override: Optional[Any],
        policy: Optional[ContextPolicySpec],
        tighten_pack_budget: bool = False,
        total_pack_budget_override: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        assembled_messages: list[dict[str, Any]] = []
        resolved_context = dict(template_context)
        context_values = dict(resolved_context.get("context", {}))
        resolved_context["context"] = context_values
        directive_budgets = self._allocate_directive_budgets(
            context_spec.messages or [], policy, total_pack_budget_override
        )
        for directive in context_spec.messages or []:
            if isinstance(directive, HistoryInsertSpec):
                assembled_messages.extend(history_messages)
                continue
            if isinstance(directive, ContextInsertSpec):
                override_budget = directive_budgets.get(id(directive))
                pack_content = self._render_pack(
                    directive.name,
                    resolved_context,
                    retriever_override,
                    override_budget or directive.budget,
                    policy,
                    tighten_pack_budget,
                    directive.weight,
                )
                context_values[directive.name] = pack_content or ""
                if pack_content:
                    assembled_messages.append({"role": "system", "content": pack_content})
                else:
                    context_values[directive.name] = ""
                continue
            if isinstance(directive, SystemMessageSpec):
                assembled_messages.append(
                    {
                        "role": "system",
                        "content": self._resolve_message_content(directive, resolved_context),
                    }
                )
                continue
            if isinstance(directive, UserMessageSpec):
                assembled_messages.append(
                    {
                        "role": "user",
                        "content": self._resolve_message_content(directive, resolved_context),
                    }
                )
                continue
            if isinstance(directive, AssistantMessageSpec):
                assembled_messages.append(
                    {
                        "role": "assistant",
                        "content": self._resolve_message_content(directive, resolved_context),
                    }
                )
                continue
        return assembled_messages

    def _render_nested_context_pack(
        self,
        context_spec: ContextDeclaration,
        template_context: dict[str, Any],
        pack_budget: Optional[Any],
        policy: Optional[ContextPolicySpec],
        tighten_pack_budget: bool,
        retriever_override: Optional[Any],
    ) -> str:
        """Render another Context declaration as a pack string."""
        nested_policy = context_spec.policy or policy
        compaction_policy = policy or nested_policy
        total_budget_override = self._extract_pack_budget_max_tokens(pack_budget, policy)

        if context_spec.messages is None:
            nested_result = self._assemble_default_with_regeneration(
                context_spec,
                base_system_prompt="",
                history_messages=[],
                user_message="",
                template_context=template_context,
                retriever_override=retriever_override,
                total_budget_override=total_budget_override,
            )
            pack_text = nested_result.system_prompt
            return self._compact_pack_text(
                pack_text, pack_budget, compaction_policy, tighten_pack_budget
            )

        if any(isinstance(directive, HistoryInsertSpec) for directive in context_spec.messages):
            raise ValueError("Nested context packs cannot include history()")

        max_iterations = 1
        if compaction_policy and getattr(compaction_policy, "max_iterations", None):
            max_iterations = max(1, int(compaction_policy.max_iterations))

        pack_scale = 1.0
        last_text = ""
        max_tokens = self._extract_pack_budget_max_tokens(pack_budget, compaction_policy)
        for _iteration in range(max_iterations):
            scaled_override = total_budget_override
            if scaled_override is not None and pack_scale < 1.0:
                scaled_override = max(1, int(scaled_override * pack_scale))

            assembled_messages = self._build_messages(
                context_spec,
                history_messages=[],
                template_context=template_context,
                retriever_override=retriever_override,
                policy=nested_policy,
                tighten_pack_budget=tighten_pack_budget or pack_scale < 1.0,
                total_pack_budget_override=scaled_override,
            )
            rendered_segments = [message.get("content", "") for message in assembled_messages]
            pack_text = self._join_nonempty(rendered_segments)
            raw_token_count = self._estimate_tokens(pack_text)
            last_text = self._compact_pack_text(
                pack_text, pack_budget, compaction_policy, tighten_pack_budget
            )

            if max_tokens is None or raw_token_count <= max_tokens:
                break
            if not compaction_policy or getattr(compaction_policy, "overflow", None) != "compact":
                break
            pack_scale *= 0.5

        return last_text

    def _compact_pack_text(
        self,
        text: str,
        pack_budget: Optional[Any],
        policy: Optional[ContextPolicySpec],
        tighten_pack_budget: bool,
    ) -> str:
        max_tokens = None
        if pack_budget is not None:
            if isinstance(pack_budget, dict):
                max_tokens = pack_budget.get("max_tokens")
            elif hasattr(pack_budget, "max_tokens"):
                max_tokens = getattr(pack_budget, "max_tokens", None)

        if max_tokens is None and policy and getattr(policy, "pack_budget", None):
            pack_budget_spec = policy.pack_budget
            max_tokens = getattr(pack_budget_spec, "default_max_tokens", None)

        if max_tokens is None:
            return text

        if tighten_pack_budget:
            max_tokens = max(1, int(max_tokens * 0.5))

        compactor = self._resolve_compactor(policy) if policy else TruncateCompactor()
        return compactor.compact(CompactionRequest(text=text, max_tokens=int(max_tokens)))

    def _split_leading_system(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split leading system messages from the rest."""
        leading_system: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        seen_non_system = False
        for message in messages:
            if message.get("role") == "system" and not seen_non_system:
                leading_system.append(message)
            else:
                seen_non_system = True
                remaining.append(message)
        return leading_system, remaining

    def _extract_user_message(
        self, messages: list[dict[str, Any]], fallback_message: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract the final user message from a list, leaving the rest as history."""
        last_user_index = None
        for idx, message in enumerate(messages):
            if message.get("role") == "user":
                last_user_index = idx
        if last_user_index is None:
            return fallback_message, messages

        user_message = messages[last_user_index].get("content", "")
        remaining = messages[:last_user_index] + messages[last_user_index + 1 :]
        return user_message, remaining

    def _join_nonempty(self, parts: Iterable[str]) -> str:
        return "\n\n".join([part for part in parts if part])
