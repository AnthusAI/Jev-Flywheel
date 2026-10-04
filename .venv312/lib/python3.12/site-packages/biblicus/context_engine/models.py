"""
Pydantic models for Biblicus Context Engine configuration.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextBudgetSpec(BaseModel):
    """
    Token budget specification for Context assembly.

    :ivar ratio: Optional ratio of the input budget to allocate.
    :vartype ratio: float or None
    :ivar max_tokens: Optional absolute token cap.
    :vartype max_tokens: int or None
    """

    model_config = ConfigDict(extra="forbid")

    ratio: Optional[float] = Field(default=None, ge=0.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_budget(self) -> "ContextBudgetSpec":
        """
        Ensure at least one budget control is provided.

        :return: Validated budget spec.
        :rtype: ContextBudgetSpec
        :raises ValueError: If neither ratio nor max_tokens is provided.
        """
        if self.ratio is None and self.max_tokens is None:
            raise ValueError("Budget must specify ratio or max_tokens")
        return self


class ContextPackBudgetSpec(BaseModel):
    """
    Default budget policy for Context packs.

    :ivar default_ratio: Optional ratio of the input budget to allocate per pack.
    :vartype default_ratio: float or None
    :ivar default_max_tokens: Optional absolute token cap per pack.
    :vartype default_max_tokens: int or None
    """

    model_config = ConfigDict(extra="forbid")

    default_ratio: Optional[float] = Field(default=None, ge=0.0)
    default_max_tokens: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_pack_budget(self) -> "ContextPackBudgetSpec":
        """
        Ensure at least one default pack budget control is provided.

        :return: Validated pack budget spec.
        :rtype: ContextPackBudgetSpec
        :raises ValueError: If neither default_ratio nor default_max_tokens is provided.
        """
        if self.default_ratio is None and self.default_max_tokens is None:
            raise ValueError("Pack budget must specify default_ratio or default_max_tokens")
        return self


class ContextExpansionSpec(BaseModel):
    """
    Pagination policy for expanding retriever packs.

    :ivar max_pages: Maximum number of retrieval pages to request.
    :vartype max_pages: int
    :ivar min_fill_ratio: Optional minimum fill ratio before stopping expansion.
    :vartype min_fill_ratio: float or None
    """

    model_config = ConfigDict(extra="forbid")

    max_pages: int = Field(default=1, ge=1)
    min_fill_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ContextPolicySpec(BaseModel):
    """
    Policy configuration for Context assembly, compaction, and expansion.

    :ivar input_budget: Optional input budget for the full assembled context.
    :vartype input_budget: ContextBudgetSpec or None
    :ivar pack_budget: Optional default budget for individual packs.
    :vartype pack_budget: ContextPackBudgetSpec or None
    :ivar overflow: Overflow behavior (for example, "compact").
    :vartype overflow: str or None
    :ivar compactor: Compactor configuration or registry key.
    :vartype compactor: str or dict[str, Any] or None
    :ivar max_iterations: Maximum compaction regeneration iterations.
    :vartype max_iterations: int or None
    :ivar expansion: Optional expansion policy for retriever pagination.
    :vartype expansion: ContextExpansionSpec or None
    """

    model_config = ConfigDict(extra="forbid")

    input_budget: Optional[ContextBudgetSpec] = None
    pack_budget: Optional[ContextPackBudgetSpec] = None
    overflow: Optional[str] = None
    compactor: Optional[Union[str, dict[str, Any]]] = None
    max_iterations: Optional[int] = None
    expansion: Optional[ContextExpansionSpec] = None


class ContextTemplateSpec(BaseModel):
    """
    Template definition for message content.

    :ivar template: Template string with dot-notation placeholders.
    :vartype template: str
    :ivar vars: Template variable overrides.
    :vartype vars: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    template: str
    vars: dict[str, Any] = Field(default_factory=dict)


class ContextMessageBase(BaseModel):
    """
    Base class for Context message directives.

    :ivar type: Directive type identifier.
    :vartype type: str
    """

    model_config = ConfigDict(extra="forbid")

    type: str


class SystemMessageSpec(ContextMessageBase):
    """
    System message directive.

    :ivar content: Literal message content.
    :vartype content: str or None
    :ivar template: Template string for message content.
    :vartype template: str or None
    :ivar vars: Template variable overrides.
    :vartype vars: dict[str, Any]
    """

    type: Literal["system"]
    content: Optional[str] = None
    template: Optional[str] = None
    vars: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content(self) -> "SystemMessageSpec":
        """
        Ensure exactly one of content/template is provided.

        :return: Validated message spec.
        :rtype: SystemMessageSpec
        :raises ValueError: If content/template usage is invalid.
        """
        if (self.content is None) == (self.template is None):
            raise ValueError("System message must define either content or template")
        return self


class UserMessageSpec(ContextMessageBase):
    """
    User message directive.

    :ivar content: Literal message content.
    :vartype content: str or None
    :ivar template: Template string for message content.
    :vartype template: str or None
    :ivar vars: Template variable overrides.
    :vartype vars: dict[str, Any]
    """

    type: Literal["user"]
    content: Optional[str] = None
    template: Optional[str] = None
    vars: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content(self) -> "UserMessageSpec":
        """
        Ensure exactly one of content/template is provided.

        :return: Validated message spec.
        :rtype: UserMessageSpec
        :raises ValueError: If content/template usage is invalid.
        """
        if (self.content is None) == (self.template is None):
            raise ValueError("User message must define either content or template")
        return self


class AssistantMessageSpec(ContextMessageBase):
    """
    Assistant message directive.

    :ivar content: Literal message content.
    :vartype content: str or None
    :ivar template: Template string for message content.
    :vartype template: str or None
    :ivar vars: Template variable overrides.
    :vartype vars: dict[str, Any]
    """

    type: Literal["assistant"]
    content: Optional[str] = None
    template: Optional[str] = None
    vars: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content(self) -> "AssistantMessageSpec":
        """
        Ensure exactly one of content/template is provided.

        :return: Validated message spec.
        :rtype: AssistantMessageSpec
        :raises ValueError: If content/template usage is invalid.
        """
        if (self.content is None) == (self.template is None):
            raise ValueError("Assistant message must define either content or template")
        return self


class ContextInsertSpec(ContextMessageBase):
    """
    Context pack insertion directive.

    :ivar name: Context pack name to insert.
    :vartype name: str
    :ivar budget: Optional pack budget override.
    :vartype budget: ContextBudgetSpec or None
    :ivar weight: Optional weight to bias pack budget allocation.
    :vartype weight: float or None
    :ivar priority: Optional priority for pack budget allocation.
    :vartype priority: int or None
    """

    type: Literal["context"]
    name: str
    budget: Optional[ContextBudgetSpec] = None
    weight: Optional[float] = None
    priority: Optional[int] = None


class HistoryInsertSpec(ContextMessageBase):
    """
    History insertion directive.

    :ivar type: Always "history".
    :vartype type: str
    """

    type: Literal["history"]


ContextMessageSpec = Union[
    SystemMessageSpec,
    UserMessageSpec,
    AssistantMessageSpec,
    ContextInsertSpec,
    HistoryInsertSpec,
]


class ContextPackSpec(BaseModel):
    """
    Context pack reference for default Context assembly.

    :ivar name: Context pack name.
    :vartype name: str
    :ivar weight: Optional weight for budget allocation.
    :vartype weight: float or None
    :ivar priority: Optional priority for budget allocation.
    :vartype priority: int or None
    :ivar budget: Optional pack budget override.
    :vartype budget: ContextBudgetSpec or None
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    weight: Optional[float] = None
    priority: Optional[int] = None
    budget: Optional[ContextBudgetSpec] = None


class ContextDeclaration(BaseModel):
    """
    Context declaration configuration.

    :ivar name: Context name.
    :vartype name: str
    :ivar policy: Optional context policy.
    :vartype policy: ContextPolicySpec or None
    :ivar messages: Optional explicit message plan.
    :vartype messages: list[ContextMessageSpec] or None
    :ivar packs: Optional default pack list.
    :vartype packs: list[ContextPackSpec] or None
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    policy: Optional[ContextPolicySpec] = None
    messages: Optional[list[ContextMessageSpec]] = None
    packs: Optional[list[ContextPackSpec]] = None

    @model_validator(mode="before")
    def _coerce_pack_entries(self) -> "ContextDeclaration":
        """
        Normalize pack entries to dicts with name fields.

        :return: Normalized context declaration.
        :rtype: ContextDeclaration
        """
        if not isinstance(self, dict):
            return self
        packs = self.get("packs")
        if packs is None:
            return self
        if isinstance(packs, str):
            self["packs"] = [{"name": packs}]
            return self
        if isinstance(packs, list):
            normalized = []
            for entry in packs:
                if isinstance(entry, str):
                    normalized.append({"name": entry})
                else:
                    normalized.append(entry)
            self["packs"] = normalized
        return self


class CorpusDeclaration(BaseModel):
    """
    Corpus declaration configuration.

    :ivar name: Corpus name.
    :vartype name: str
    :ivar config: Corpus configuration payload.
    :vartype config: dict[str, Any]
    """

    model_config = ConfigDict(extra="allow")

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class RetrieverDeclaration(BaseModel):
    """
    Retriever declaration configuration.

    :ivar name: Retriever name.
    :vartype name: str
    :ivar corpus: Optional corpus identifier.
    :vartype corpus: str or None
    :ivar config: Retriever configuration payload.
    :vartype config: dict[str, Any]
    """

    model_config = ConfigDict(extra="allow")

    name: str
    corpus: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)


class CompactorDeclaration(BaseModel):
    """
    Compactor declaration configuration.

    :ivar name: Compactor name.
    :vartype name: str
    :ivar config: Compactor configuration payload.
    :vartype config: dict[str, Any]
    """

    model_config = ConfigDict(extra="allow")

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class ContextRetrieverRequest(BaseModel):
    """
    Retrieval request for Context packs.

    :ivar query: Query text issued against the retriever.
    :vartype query: str
    :ivar offset: Offset into the ranked candidate list.
    :vartype offset: int
    :ivar limit: Maximum number of items to return.
    :vartype limit: int
    :ivar maximum_total_characters: Optional maximum total characters for the pack.
    :vartype maximum_total_characters: int or None
    :ivar max_tokens: Optional maximum token budget for the pack.
    :vartype max_tokens: int or None
    :ivar metadata: Optional metadata for retriever implementations.
    :vartype metadata: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=3, ge=1)
    maximum_total_characters: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
