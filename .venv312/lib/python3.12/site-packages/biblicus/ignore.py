"""
Corpus ignore rules for bulk import and crawling.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class CorpusIgnoreSpec(BaseModel):
    """
    Parsed corpus ignore patterns.

    Patterns are matched against a forward-slash relative path string.

    :ivar patterns: Glob patterns to ignore.
    :vartype patterns: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    patterns: List[str] = Field(default_factory=list)

    def matches(self, relpath: str) -> bool:
        """
        Return True if the relative path matches any ignore pattern.

        :param relpath: Forward-slash relative path.
        :type relpath: str
        :return: True if the path should be ignored.
        :rtype: bool
        """
        normalized = relpath.replace("\\", "/").lstrip("/")
        return any(fnmatch.fnmatch(normalized, pattern) for pattern in self.patterns)


def load_corpus_ignore_spec(corpus_root: Path) -> CorpusIgnoreSpec:
    """
    Load ignore patterns from the corpus ignore file, if present.

    The ignore file is stored at the corpus root as `.biblicusignore`.

    :param corpus_root: Corpus root directory.
    :type corpus_root: Path
    :return: Parsed ignore specification.
    :rtype: CorpusIgnoreSpec
    """
    ignore_path = corpus_root / ".biblicusignore"
    if not ignore_path.is_file():
        return CorpusIgnoreSpec(patterns=[])

    patterns: List[str] = []
    for raw_line in ignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        patterns.append(line)
    return CorpusIgnoreSpec(patterns=patterns)
