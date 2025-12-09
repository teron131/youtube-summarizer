"""
This module contains helper functions for string manipulation, data parsing, and logging, used across the YouTube Summarizer application.
"""

import json
import re
import sys
from typing import Any, Union

from opencc import OpenCC
from pydantic import BaseModel

_OPENCC_S2HK = OpenCC("s2hk")
YOUTUBE_URL_PATTERN = re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+")
YOUTUBE_ID_PATTERN = re.compile(r"(?:v=|youtu\.be/)([\w-]+)")


def log_and_print(message: str):
    """Log and print message to ensure visibility in Railway."""
    print(message, flush=True)
    sys.stdout.flush()


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


def _type_string(spec: dict[str, Any], defs: dict[str, Any]) -> tuple[str, set[str]]:
    """Return a readable type string and any referenced definitions."""
    ref = spec.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        name = ref.split("/")[-1]
        return name, {name}

    if "anyOf" in spec:
        types = []
        refs = set()
        for option in spec["anyOf"]:
            if option.get("type") == "null":
                continue
            type_str, type_refs = _type_string(option, defs)
            types.append(type_str)
            refs |= type_refs
        return (" | ".join(sorted(set(types))) if types else "Any"), refs

    if spec.get("type") == "array":
        item_type, item_refs = _type_string(spec.get("items", {}), defs)
        return f"list[{item_type}]", item_refs

    return _simple_type(spec), set()


def _parse_properties(properties: dict[str, Any], defs: dict[str, Any]) -> tuple[list[str], set[str]]:
    lines = []
    refs = set()
    for name, spec in properties.items():
        type_str, type_refs = _type_string(spec, defs)
        refs |= type_refs
        desc = spec.get("description")
        lines.append(f"{name}: {type_str}" + (f" = {desc}" if desc else ""))
    return lines, refs


def schema_to_string(schema: Union[dict[str, Any], BaseModel]) -> str:
    """Flatten a Pydantic BaseModel or JSON schema into a readable string."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    elif isinstance(schema, BaseModel):
        schema = schema.model_json_schema()

    lines = []
    defs = schema.get("$defs", {})

    main_lines, queued = _parse_properties(schema.get("properties", {}), defs)
    lines.extend(main_lines)

    seen = set()
    while queued:
        name = queued.pop()
        if name in seen or name not in defs or defs[name].get("type") != "object":
            continue
        seen.add(name)

        lines.extend(["", f"## {name} Type", ""])
        def_lines, new_refs = _parse_properties(defs[name].get("properties", {}), defs)
        lines.extend(f"  {line}" for line in def_lines)
        queued |= new_refs - seen

    return "\n".join(lines)


# Module-level compiled patterns for maximum performance
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespace and newlines."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube watch or short link."""
    return bool(YOUTUBE_URL_PATTERN.match(url))


def _extract_video_id(url: str) -> str | None:
    match = YOUTUBE_ID_PATTERN.search(url)
    return match.group(1) if match else None


def clean_youtube_url(url: str) -> str:
    """Normalize YouTube URLs to https://www.youtube.com/watch?v=<id> when possible."""
    video_id = _extract_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else url


def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Hong Kong variant)."""
    return _OPENCC_S2HK.convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result (JSON) to plain text."""
    chunks = result.get("chunks") or []
    lines = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text")]
    return s2hk("\n".join(lines))


def parse_youtube_json_captions(json_content: str) -> str:
    """
    Parse YouTube's JSON timedtext format and extract plain text.
    Returns the original content if parsing fails.
    """
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return json_content

    events = data.get("events") or []
    text_parts = [seg["utf8"] for event in events for seg in event.get("segs", []) if "utf8" in seg]
    return "".join(text_parts).strip() or json_content


def srt_to_txt(srt_content: str) -> str:
    """Convert SRT (SubRip Text) format content to plain text."""
    lines = [line.strip() for line in srt_content.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line]
    return s2hk("\n".join(lines))
