"""General-purpose utilities for line-tagged content processing.

These utilities support adding, removing, and filtering content using line-level
tags (e.g., [L1], [L2]). This is useful for LLM workflows where models identify
specific sections of text by their line numbers.
"""

import re

from pydantic import BaseModel, Field


class TagRange(BaseModel):
    """Represents a range of lines to be removed from the content."""

    start_tag: str = Field(description="The starting line tag, e.g., [L10]")
    end_tag: str = Field(description="The ending line tag, e.g., [L20]")


def tag_content(text: str) -> str:
    """Prepend [LX] tags to each line of the content."""
    lines = text.splitlines()
    return "\n".join(f"[L{i + 1}] {line}" for i, line in enumerate(lines))


def untag_content(text: str) -> str:
    """Remove [LX] tags from the content."""
    # Use re.MULTILINE to match at the start of each line
    return re.sub(r"^\[L\d+\]\s*", "", text, flags=re.MULTILINE)


def filter_content(tagged_text: str, ranges: list[TagRange]) -> str:
    """Remove lines between start_tag and end_tag from the tagged content."""
    lines = tagged_text.splitlines()
    if not lines or not ranges:
        return tagged_text

    # 1. Build tag mapping efficiently
    tag_to_idx = {}
    for i, line in enumerate(lines):
        if line.startswith("[L"):
            end_bracket = line.find("]")
            if end_bracket != -1:
                tag_to_idx[line[: end_bracket + 1]] = i

    # 2. Boolean mask (initialized to True = keep)
    keep_mask = [True] * len(lines)

    # 3. Mark ranges to remove
    for r in ranges:
        s = tag_to_idx.get(r.start_tag)
        e = tag_to_idx.get(r.end_tag)
        if s is not None and e is not None:
            # Ensure correct ordering and inclusive range
            start, end = (s, e) if s <= e else (e, s)
            # Slice assignment is efficient in Python
            length = end - start + 1
            if length > 0:
                keep_mask[start : end + 1] = [False] * length

    # 4. Filter and join
    return "\n".join(line for line, keep in zip(lines, keep_mask, strict=True) if keep)
