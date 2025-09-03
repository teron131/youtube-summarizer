"""
This module contains helper functions for string manipulation, data parsing, and logging, used across the YouTube Summarizer application.
"""

import json
import re
import sys
from typing import Any, Union

from opencc import OpenCC
from pydantic import BaseModel


def log_and_print(message: str):
    """Log and print message to ensure visibility in Railway."""
    print(message, flush=True)
    sys.stdout.flush()


def schema_to_string(schema: Union[dict[str, Any], BaseModel]) -> str:
    """Parse a Pydantic BaseModel or a JSON schema and return a string representation of the schema.

    This provides context to LLM and avoids JSON format causing LangChain errors."""

    def _parse_properties(
        properties: dict[str, Any],
        required_fields: list[str],
        defs: dict[str, Any],
    ) -> tuple[list[str], set[str]]:
        lines = []
        refs = set()

        for name, spec in properties.items():
            type_str, type_refs = _type_string(spec, defs)
            refs |= type_refs
            desc = spec.get("description")
            lines.append(f"{name}: {type_str}" + (f" = {desc}" if desc else ""))

        return lines, refs

    def _type_string(spec: dict[str, Any], defs: dict[str, Any]) -> tuple[str, set[str]]:
        # $ref
        if "$ref" in spec:
            ref = spec["$ref"]
            if ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                return name, {name}

        # anyOf
        if "anyOf" in spec:
            types = []
            refs = set()
            for opt in spec["anyOf"]:
                if opt.get("type") == "null":
                    continue
                type_str, type_refs = _type_string(opt, defs)
                types.append(type_str)
                refs |= type_refs
            return (" | ".join(sorted(set(types))) if types else "Any"), refs

        # arrays
        if spec.get("type") == "array":
            item = spec.get("items", {})
            type_str, type_refs = _type_string(item, defs)
            return f"list[{type_str}]", type_refs

        # simple types
        return _simple_type(spec), set()

    def _simple_type(spec: dict[str, Any]) -> str:
        type_mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "object": "dict",
            "array": "list[Any]",
        }
        return type_mapping.get(spec.get("type", "Any"), spec.get("type", "Any"))

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    elif isinstance(schema, BaseModel):
        schema = schema.model_json_schema()

    lines = []
    defs = schema.get("$defs", {})

    main_lines, queued = _parse_properties(schema.get("properties", {}), schema.get("required", []), defs)
    lines.extend(main_lines)

    seen = set()
    while queued:
        name = queued.pop()
        if name in seen or name not in defs or defs[name].get("type") != "object":
            continue
        seen.add(name)

        lines.extend(["", f"## {name} Type", ""])
        def_lines, new_refs = _parse_properties(defs[name].get("properties", {}), defs[name].get("required", []), defs)
        lines.extend(f"  {line}" for line in def_lines)
        queued |= new_refs - seen

    return "\n".join(lines)


# Module-level compiled patterns for maximum performance
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespace and newlines."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_youtube_url(url: str) -> bool:
    """
    Check if the URL is a valid YouTube URL.
    Accepts both youtube.com/watch?v= and youtu.be/ formats.
    """
    youtube_patterns = [
        r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+",
        r"https?://(?:www\.)?youtu\.be/[\w-]+",
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def clean_youtube_url(url: str) -> str:
    """
    Clean the YouTube URL by extracting video ID and removing extra parameters.
    Converts both formats to standard youtube.com/watch?v=ID format.
    """
    # Extract video ID from youtube.com/watch?v=ID format
    youtube_match = re.search(r"youtube\.com/watch\?v=([\w-]+)", url)
    if youtube_match:
        video_id = youtube_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"

    # Extract video ID from youtu.be/ID format
    youtu_be_match = re.search(r"youtu\.be/([\w-]+)", url)
    if youtu_be_match:
        video_id = youtu_be_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"

    # Return original URL if no match found
    return url


def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Hong Kong variant)."""
    return OpenCC("s2hk").convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result (JSON) to plain text."""
    txt_content = "\n".join(chunk["text"].strip() for chunk in result.get("chunks", []))
    return s2hk(txt_content)


def parse_youtube_json_captions(json_content: str) -> str:
    """
    Parse YouTube's JSON timedtext format and extract plain text.
    Handles the specific structure of YouTube's auto-generated captions.
    """
    try:
        data = json.loads(json_content)
        text_parts = []
        if "events" in data:
            for event in data["events"]:
                if "segs" in event:
                    for seg in event["segs"]:
                        if "utf8" in seg:
                            text_parts.append(seg["utf8"])
        full_text = "".join(text_parts)
        return full_text.strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, return the original content
        return json_content


def srt_to_txt(srt_content: str) -> str:
    """Convert SRT (SubRip Text) format content to plain text."""
    lines = []
    for line in srt_content.splitlines():
        line = line.strip()
        # Filter out timestamp lines and sequence numbers
        if line and not line.isdigit() and "-->" not in line:
            lines.append(line)
    return s2hk("\n".join(lines))
