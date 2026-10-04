"""
Website crawl utilities for Biblicus corpora.
"""

from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from typing import Deque, List, Optional, Set
from urllib.parse import urldefrag, urljoin

from pydantic import BaseModel, ConfigDict, Field

from .ignore import load_corpus_ignore_spec
from .sources import load_source


class CrawlRequest(BaseModel):
    """
    Request describing a website crawl into a corpus.

    :ivar root_url: Initial uniform resource locator to fetch.
    :vartype root_url: str
    :ivar allowed_prefix: Uniform resource locator prefix that limits which links are eligible for crawl.
    :vartype allowed_prefix: str
    :ivar max_items: Maximum number of items to store during the crawl.
    :vartype max_items: int
    :ivar tags: Tags to apply to stored items.
    :vartype tags: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    root_url: str = Field(min_length=1)
    allowed_prefix: str = Field(min_length=1)
    max_items: int = Field(default=50, ge=1)
    tags: List[str] = Field(default_factory=list)


class CrawlResult(BaseModel):
    """
    Summary result for a crawl execution.

    :ivar crawl_id: Crawl identifier used in the corpus raw import namespace.
    :vartype crawl_id: str
    :ivar discovered_items: Total number of distinct uniform resource locators discovered.
    :vartype discovered_items: int
    :ivar fetched_items: Number of eligible items fetched over hypertext transfer protocol.
    :vartype fetched_items: int
    :ivar stored_items: Number of items stored into the corpus.
    :vartype stored_items: int
    :ivar skipped_outside_prefix_items: Number of discovered items outside the allowed prefix.
    :vartype skipped_outside_prefix_items: int
    :ivar skipped_ignored_items: Number of eligible items skipped due to corpus ignore rules.
    :vartype skipped_ignored_items: int
    :ivar errored_items: Number of eligible items that failed to fetch or store.
    :vartype errored_items: int
    """

    model_config = ConfigDict(extra="forbid")

    crawl_id: str
    discovered_items: int = Field(default=0, ge=0)
    fetched_items: int = Field(default=0, ge=0)
    stored_items: int = Field(default=0, ge=0)
    skipped_outside_prefix_items: int = Field(default=0, ge=0)
    skipped_ignored_items: int = Field(default=0, ge=0)
    errored_items: int = Field(default=0, ge=0)


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs):  # type: ignore[no-untyped-def]
        _ = tag
        for key, value in attrs:
            if key in {"href", "src"} and isinstance(value, str) and value.strip():
                self.links.append(value.strip())


def _normalize_crawl_url(candidate: str, *, base_url: str) -> Optional[str]:
    joined = urljoin(base_url, candidate)
    joined, _fragment = urldefrag(joined)
    joined = joined.strip()
    if joined.startswith(("mailto:", "javascript:")):
        return None
    return joined


def _crawl_relative_path(url: str, *, allowed_prefix: str) -> str:
    relative = url[len(allowed_prefix) :].lstrip("/")
    if not relative or relative.endswith("/"):
        relative = relative.rstrip("/") + "/index.html" if relative else "index.html"
    return relative


def _should_parse_links(media_type: str) -> bool:
    return media_type.startswith("text/html")


def _discover_links(html_text: str, *, base_url: str) -> List[str]:
    parser = _LinkExtractor()
    parser.feed(html_text)
    discovered: List[str] = []
    for raw in parser.links:
        normalized = _normalize_crawl_url(raw, base_url=base_url)
        if normalized is not None:
            discovered.append(normalized)
    return discovered


def crawl_into_corpus(*, corpus, request: CrawlRequest) -> CrawlResult:  # type: ignore[no-untyped-def]
    """
    Crawl a website prefix into a corpus.

    :param corpus: Target corpus to receive crawled items.
    :type corpus: biblicus.corpus.Corpus
    :param request: Crawl request describing limits and allowed prefix.
    :type request: CrawlRequest
    :return: Crawl result summary.
    :rtype: CrawlResult
    """
    ignore_spec = load_corpus_ignore_spec(corpus.root)
    allowed_prefix = request.allowed_prefix
    root_url = request.root_url

    crawl_id = corpus.create_crawl_id()

    queue: Deque[str] = deque([root_url])
    seen: Set[str] = set()
    stored_count = 0
    fetched_count = 0
    skipped_outside_prefix_count = 0
    skipped_ignored_count = 0
    errored_count = 0
    discovered_urls: Set[str] = set()

    while queue and stored_count < request.max_items:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        discovered_urls.add(url)

        if not url.startswith(allowed_prefix):
            skipped_outside_prefix_count += 1
            continue

        relative_path = _crawl_relative_path(url, allowed_prefix=allowed_prefix)
        if ignore_spec.matches(relative_path):
            skipped_ignored_count += 1
            continue

        try:
            payload = load_source(url)
            fetched_count += 1
            corpus.ingest_crawled_payload(
                crawl_id=crawl_id,
                relative_path=relative_path,
                data=payload.data,
                filename=payload.filename,
                media_type=payload.media_type,
                source_uri=payload.source_uri,
                tags=request.tags,
            )
            stored_count += 1
        except Exception:
            errored_count += 1
            continue

        if _should_parse_links(payload.media_type):
            text = payload.data.decode("utf-8", errors="replace")
            for discovered in _discover_links(text, base_url=url):
                queue.append(discovered)

    return CrawlResult(
        crawl_id=crawl_id,
        discovered_items=len(discovered_urls),
        fetched_items=fetched_count,
        stored_items=stored_count,
        skipped_outside_prefix_items=skipped_outside_prefix_count,
        skipped_ignored_items=skipped_ignored_count,
        errored_items=errored_count,
    )
