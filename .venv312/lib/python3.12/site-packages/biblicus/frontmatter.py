"""
Markdown front matter helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class FrontMatterDocument:
    """
    Parsed front matter and markdown body.

    :ivar metadata: Front matter metadata mapping.
    :vartype metadata: dict[str, Any]
    :ivar body: Markdown body text.
    :vartype body: str
    """

    metadata: Dict[str, Any]
    body: str


def parse_front_matter(text: str) -> FrontMatterDocument:
    """
    Parse Yet Another Markup Language front matter from a Markdown document.

    :param text: Markdown content with optional front matter.
    :type text: str
    :return: Parsed front matter and body.
    :rtype: FrontMatterDocument
    :raises ValueError: If front matter is present but not a mapping.
    """
    if not text.startswith("---\n"):
        return FrontMatterDocument(metadata={}, body=text)

    front_matter_end = text.find("\n---\n", 4)
    if front_matter_end == -1:
        return FrontMatterDocument(metadata={}, body=text)

    raw_yaml = text[4:front_matter_end]
    body = text[front_matter_end + len("\n---\n") :]
    if body.startswith("\n"):
        body = body[1:]

    metadata = yaml.safe_load(raw_yaml) or {}
    if not isinstance(metadata, dict):
        raise ValueError("Yet Another Markup Language front matter must be a mapping object")

    return FrontMatterDocument(metadata=dict(metadata), body=body)


def render_front_matter(metadata: Dict[str, Any], body: str) -> str:
    """
    Render Yet Another Markup Language front matter with a Markdown body.

    :param metadata: Front matter metadata mapping.
    :type metadata: dict[str, Any]
    :param body: Markdown body text.
    :type body: str
    :return: Markdown with Yet Another Markup Language front matter.
    :rtype: str
    """
    if not metadata:
        return body

    yaml_text = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()

    return f"---\n{yaml_text}\n---\n{body}"


def split_markdown_front_matter(path_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Split Markdown into front matter metadata and body.

    :param path_text: Markdown content.
    :type path_text: str
    :return: Metadata mapping and body text.
    :rtype: tuple[dict[str, Any], str]
    """
    parsed_document = parse_front_matter(path_text)
    return parsed_document.metadata, parsed_document.body
