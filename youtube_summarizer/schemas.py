from pydantic import BaseModel, Field, field_validator

from .utils import s2hk


class Chapter(BaseModel):
    """Represents a single chapter in the video summary."""

    title: str = Field(
        description="A concise chapter heading.",
    )
    description: str = Field(
        description="A substantive chapter description grounded in the content. Include key facts (numbers/names/steps) when present. Avoid meta-language like 'the video', 'the author', 'the speaker says'—state the content directly.",
    )
    start_time: str | None = Field(
        None,
        description="Optional chapter start timestamp in the format MM:SS.",
    )
    end_time: str | None = Field(
        None,
        description="Optional chapter end timestamp matching the same format as start_time.",
    )

    @field_validator("title", "description")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese if needed (helper hook)."""
        return s2hk(value)


class Summary(BaseModel):
    """Complete analysis of a YouTube video."""

    overview: str = Field(
        description="An end-to-end summary of the whole content (main thesis + arc), written in direct statements without meta-language.",
    )
    chapters: list[Chapter] = Field(
        min_length=1,
        description="Chronological, non-overlapping chapters covering the core content.",
    )

    @field_validator("overview")
    def convert_string_to_hk(cls, value: str) -> str:
        """Convert string fields to Traditional Chinese."""
        return s2hk(value)

    def to_text(self) -> str:
        """Format the summary as a readable string."""
        lines = [
            "=" * 80,
            "SUMMARY:",
            "=" * 80,
            f"\nOverview:\n{self.overview}",
            f"\nChapters ({len(self.chapters)}):",
        ]

        for i, chapter in enumerate(self.chapters, 1):
            lines.append(f"\n  Chapter {i}: {chapter.title}")
            lines.append(f"    Summary: {chapter.description}")
            if chapter.start_time or chapter.end_time:
                time_range = f"{chapter.start_time or '?'} - {chapter.end_time or '?'}"
                lines.append(f"    Time: {time_range}")

        return "\n".join(lines)
