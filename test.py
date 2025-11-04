"""
Test script using Scrape Creators API + OpenRouter with LangGraph verification workflow.

Environment Variables Required:
    - SCRAPECREATORS_API_KEY: Your Scrape Creators API key
    - OPENROUTER_API_KEY: Your OpenRouter API key

Usage:
    uv run python test.py [youtube_url] [model_name]

Examples:
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "x-ai/grok-4-fast"
"""

import os
import sys
import time
from typing import Any, Literal, Optional, Union

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from youtube_summarizer.utils import schema_to_string

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Configuration for the test workflow."""

    ANALYSIS_MODEL = "x-ai/grok-4-fast"
    QUALITY_MODEL = "x-ai/grok-4-fast"
    MIN_QUALITY_SCORE = 90
    MAX_ITERATIONS = 2


# ============================================================================
# Data Models
# ============================================================================


class Analysis(BaseModel):
    """Analysis with essential fields."""

    summary: str = Field(description="A comprehensive summary of the video content (150-400 words)")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience", min_length=3, max_length=8)
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned", min_length=3, max_length=6)


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect (Fail=poor, Refine=adequate, Pass=excellent)")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment."""

    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    accuracy: Rate = Field(description="Rate for accuracy: Content directly supported by transcript (no external additions)")
    structure: Rate = Field(description="Rate for structure: Summary, takeaways, and key_facts are properly formatted")
    grammar: Rate = Field(description="Rate for grammar: No typos, grammatical mistakes, appropriate wordings")
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")

    @property
    def all_aspects(self) -> list[Rate]:
        """Return all quality aspects as a list."""
        return [self.completeness, self.accuracy, self.structure, self.grammar, self.no_garbage]

    @property
    def total_score(self) -> int:
        """Calculate total score based on all quality aspects."""
        score_map = {"Fail": 0, "Refine": 1, "Pass": 2}
        return sum(score_map[aspect.rate] for aspect in self.all_aspects)

    @property
    def max_possible_score(self) -> int:
        """Calculate maximum possible score (all aspects = Pass)."""
        return len(self.all_aspects) * 2

    @property
    def percentage_score(self) -> int:
        """Calculate percentage score out of 100."""
        return int((self.total_score / self.max_possible_score) * 100)

    @property
    def is_acceptable(self) -> bool:
        """Whether the analysis meets quality standards."""
        return self.percentage_score >= Config.MIN_QUALITY_SCORE


class GraphInput(BaseModel):
    """Input for the summarization workflow."""

    transcript: str
    analysis_model: str = Field(default=Config.ANALYSIS_MODEL)
    quality_model: str = Field(default=Config.QUALITY_MODEL)


class GraphOutput(BaseModel):
    """Output from the summarization workflow."""

    analysis: Analysis
    quality: Quality
    iteration_count: int


class GraphState(BaseModel):
    """State for the summarization workflow."""

    transcript: str
    analysis_model: str = Field(default=Config.ANALYSIS_MODEL)
    quality_model: str = Field(default=Config.QUALITY_MODEL)
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


# ============================================================================
# Prompt Builder
# ============================================================================


class PromptBuilder:
    """Centralized prompt builder for analysis."""

    @staticmethod
    def _extract_field_info(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
        """Extract field descriptions and constraints from a Pydantic model."""
        schema = model.model_json_schema()
        fields_info = {}

        for field_name, field_info in schema.get("properties", {}).items():
            fields_info[field_name] = {
                "description": field_info.get("description", ""),
                "type": field_info.get("type"),
                "min_length": field_info.get("minItems"),  # JSON schema uses minItems
                "max_length": field_info.get("maxItems"),  # JSON schema uses maxItems
                "required": field_name in schema.get("required", []),
            }

        return fields_info

    @staticmethod
    def _build_length_summary(fields_info: dict[str, dict[str, Any]]) -> str:
        """Build a concise length summary from field constraints."""
        parts = []
        for field_name, info in fields_info.items():
            display_name = field_name.replace("_", " ").title()
            min_length = info.get("min_length")
            max_length = info.get("max_length")

            if field_name == "summary":
                # Extract word count from description if available
                desc = info.get("description", "")
                if "150-400 words" in desc or "(150-400 words)" in desc:
                    parts.append(f"Summary (150-400 words)")
                else:
                    parts.append(f"Summary")
            elif min_length is not None and max_length is not None:
                parts.append(f"{display_name} ({min_length}-{max_length} items)")

        return ", ".join(parts)

    @staticmethod
    def _build_length_guidelines(fields_info: dict[str, dict[str, Any]]) -> list[str]:
        """Build length guidelines list from field constraints."""
        lines = []
        for field_name, info in fields_info.items():
            display_name = field_name.replace("_", " ").title()
            min_length = info.get("min_length")
            max_length = info.get("max_length")

            if field_name == "summary":
                # Extract word count from description if available
                desc = info.get("description", "")
                if "150-400 words" in desc or "(150-400 words)" in desc:
                    lines.append("- Summary: 150-400 words")
                else:
                    lines.append("- Summary: As specified in description")
            elif min_length is not None and max_length is not None:
                lines.append(f"- {display_name}: {min_length}-{max_length} items")

        return lines

    @staticmethod
    def build_analysis_prompt() -> str:
        """Build analysis prompt from Analysis model."""
        schema = schema_to_string(Analysis)
        fields_info = PromptBuilder._extract_field_info(Analysis)

        field_requirements = []
        for field_name, info in fields_info.items():
            if not info["description"]:
                continue
            display_name = field_name.replace("_", " ").title()
            requirement = f"- {display_name}: {info['description']}"
            min_length = info.get("min_length")
            max_length = info.get("max_length")
            if min_length is not None and max_length is not None:
                requirement += f" ({min_length}-{max_length} items)"
            field_requirements.append(requirement)

        prompt_parts = [
            "Create a comprehensive analysis that strictly follows the transcript content.",
            "",
            "OUTPUT SCHEMA:",
            schema,
            "",
            "FIELD REQUIREMENTS:",
            "\n".join(field_requirements),
            "",
            "CORE REQUIREMENTS:",
            "- ACCURACY: Every claim must be directly supported by the transcript",
            "- TONE: Write in objective, article-like style (avoid 'This video...', 'The speaker...')",
            "- AVOID META-DESCRIPTIVE LANGUAGE: Do not use phrases like 'This analysis explores', etc. Write direct, factual content only",
            "",
            "CONTENT FILTERING:",
            "- Remove all promotional content (speaker intros, calls-to-action, self-promotion)",
            "- Keep only educational content",
            "- Correct obvious typos naturally",
            "",
            "QUALITY CHECKS:",
            "- Content matches transcript exactly (no external additions)",
            "- All promotional content removed",
            "- Typos corrected naturally, meaning preserved",
            f"- Length balanced: {PromptBuilder._build_length_summary(fields_info)}",
        ]

        return "\n".join(prompt_parts)

    @staticmethod
    def build_quality_prompt() -> str:
        """Build quality evaluation prompt from Quality model."""
        schema = schema_to_string(Quality)
        fields_info = PromptBuilder._extract_field_info(Quality)

        aspects_lines = []
        for idx, (field_name, info) in enumerate(fields_info.items(), 1):
            aspect_name = field_name.upper().replace("_", " ")
            desc = info["description"]
            description = desc.split(":", 1)[1].strip() if ":" in desc else desc
            aspects_lines.append(f"{idx}. {aspect_name}: {description}")

        # Build length guidelines from Analysis model
        analysis_fields = PromptBuilder._extract_field_info(Analysis)
        length_lines = PromptBuilder._build_length_guidelines(analysis_fields)

        prompt_parts = [
            "Evaluate the analysis on the following aspects. Rate each 'Fail', 'Refine', or 'Pass' with a specific reason.",
            "",
            "ASPECTS:",
            "\n".join(aspects_lines),
            "",
            "LENGTH GUIDELINES:",
            "\n".join(length_lines),
            "",
            "QUALITY STANDARDS:",
            "- Transcript-based content only",
            "- Professional article-like tone",
            "",
            "Provide specific rates and reasons for each aspect.",
            "",
            "OUTPUT SCHEMA:",
            schema,
        ]

        return "\n".join(prompt_parts)

    @staticmethod
    def build_improvement_prompt() -> str:
        """Build improvement prompt."""
        schema = schema_to_string(Analysis)
        fields_info = PromptBuilder._extract_field_info(Analysis)

        field_requirements = []
        for field_name, info in fields_info.items():
            if not info["description"]:
                continue
            display_name = field_name.replace("_", " ").title()
            requirement = f"- {display_name}: {info['description']}"
            min_length = info.get("min_length")
            max_length = info.get("max_length")
            if min_length is not None and max_length is not None:
                requirement += f" ({min_length}-{max_length} items)"
            field_requirements.append(requirement)

        prompt_parts = [
            "Improve the analysis based on quality feedback while maintaining transcript accuracy.",
            "",
            "IMPROVEMENT PRIORITIES:",
            "1. TRANSCRIPT ACCURACY: All content must be transcript-supported",
            "2. PROMOTIONAL REMOVAL: Remove all intros, calls-to-action, self-promotion",
            "3. WRITING STYLE: Use objective, article-like tone",
            "4. AVOID META-DESCRIPTIVE LANGUAGE: Remove phrases like 'This analysis explores', etc.",
            "5. TYPO CORRECTION: Fix obvious typos naturally",
            "6. ARRAY FORMATTING: Return takeaways/key_facts as simple string arrays",
            "",
            "CONTENT TARGETS:",
            "\n".join(field_requirements),
            "",
            "OUTPUT SCHEMA:",
            schema,
        ]

        return "\n".join(prompt_parts)


# ============================================================================
# Model Client
# ============================================================================


def create_openrouter_llm(model: str) -> BaseChatModel:
    """Create OpenRouter LLM instance."""
    return init_chat_model(
        model=model,
        model_provider="openai",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )


# ============================================================================
# Utilities
# ============================================================================


class QualityUtils:
    """Utility functions for quality operations."""

    @staticmethod
    def print_quality_breakdown(quality: Quality) -> None:
        """Print quality breakdown with all aspects."""
        print("📈 Quality breakdown:")
        print(f"Completeness: {quality.completeness.rate} - {quality.completeness.reason}")
        print(f"Accuracy: {quality.accuracy.rate} - {quality.accuracy.reason}")
        print(f"Structure: {quality.structure.rate} - {quality.structure.reason}")
        print(f"Grammar: {quality.grammar.rate} - {quality.grammar.reason}")
        print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
        print(f"Total Score: {quality.total_score}/{quality.max_possible_score} ({quality.percentage_score}%)")

        if not quality.is_acceptable:
            print(f"⚠️  Quality below threshold ({Config.MIN_QUALITY_SCORE}%), refinement needed")


class PromptUtils:
    """Utility functions for prompt operations."""

    @staticmethod
    def create_improvement_context(analysis: Analysis, quality: Quality) -> str:
        """Create improvement prompt context from analysis and quality feedback."""
        return f"""# Improve this video analysis based on the following feedback:

## Analysis:
{repr(analysis)}

## Quality Assessment:
{repr(quality)}

Please provide an improved version that addresses the specific issues identified above to improve the overall quality score."""


# ============================================================================
# Analysis Node
# ============================================================================


def analysis_node(state: GraphState) -> dict[str, Union[Analysis, int]]:
    """Generate or refine analysis using OpenRouter."""
    llm = create_openrouter_llm(state.analysis_model)
    structured_llm = llm.with_structured_output(Analysis)

    # Refinement path when previous quality feedback exists
    if state.quality is not None and state.analysis is not None:
        print("🔧 Refining analysis based on quality feedback...")
        improvement_context = PromptUtils.create_improvement_context(state.analysis, state.quality)
        improvement_system_prompt = PromptBuilder.build_improvement_prompt()

        transcript_context = f"Original Transcript:\n{state.transcript}"
        full_improvement_prompt = f"{transcript_context}\n\n{improvement_context}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", improvement_system_prompt),
                ("human", "{improvement_prompt}"),
            ]
        )
        chain = prompt | structured_llm

        try:
            result: Analysis = chain.invoke({"improvement_prompt": full_improvement_prompt})
            print("✨ Analysis refined successfully")
            return {"analysis": result, "iteration_count": state.iteration_count + 1}
        except Exception as e:
            print(f"❌ Refinement failed: {str(e)}")
            raise RuntimeError(f"Refinement failed: {str(e)}") from e

    # Generation path
    print(f"📝 Generating initial analysis. Transcript length: {len(state.transcript)} characters")
    analysis_prompt = PromptBuilder.build_analysis_prompt()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", analysis_prompt),
            ("human", "{content}"),
        ]
    )
    chain = prompt | structured_llm

    try:
        result: Analysis = chain.invoke({"content": state.transcript})
        print("📊 Analysis completed")
        return {"analysis": result, "iteration_count": state.iteration_count + 1}
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise RuntimeError(f"Analysis failed: {str(e)}") from e


# ============================================================================
# Quality Node
# ============================================================================


def quality_node(state: GraphState) -> dict[str, Union[Quality, bool]]:
    """Check the quality of the generated analysis."""
    print("🔍 Performing quality check...")
    print(f"🔍 Using model: {state.quality_model}")

    llm = create_openrouter_llm(state.quality_model)
    structured_llm = llm.with_structured_output(Quality)

    quality_prompt = PromptBuilder.build_quality_prompt()
    analysis_text = repr(state.analysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", quality_prompt),
            ("human", "{analysis_text}"),
        ]
    )
    chain = prompt | structured_llm

    try:
        quality: Quality = chain.invoke({"analysis_text": analysis_text})
        QualityUtils.print_quality_breakdown(quality)

        return {
            "quality": quality,
            "is_complete": quality.percentage_score >= Config.MIN_QUALITY_SCORE or state.iteration_count >= Config.MAX_ITERATIONS,
        }
    except Exception as e:
        print(f"❌ Quality check failed: {str(e)}")
        raise RuntimeError(f"Quality check failed: {str(e)}") from e


# ============================================================================
# Graph Workflow
# ============================================================================


def should_continue(state: GraphState) -> str:
    """Determine next step in workflow."""
    if state.is_complete:
        print(f"🔄 Workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < Config.MAX_ITERATIONS:
        print(f"🔄 Quality {state.quality.percentage_score}% below threshold {Config.MIN_QUALITY_SCORE}%, refining (iteration {state.iteration_count + 1})")
        return "analysis"
    else:
        print(f"🔄 Workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph."""
    builder = StateGraph(GraphState)

    builder.add_node("analysis", analysis_node)
    builder.add_node("quality", quality_node)

    builder.add_edge(START, "analysis")
    builder.add_edge("analysis", "quality")

    builder.add_conditional_edges(
        "quality",
        should_continue,
        {
            "analysis": "analysis",
            END: END,
        },
    )

    return builder


def create_compiled_graph():
    """Create and compile the summarization graph."""
    return create_summarization_graph().compile()


# ============================================================================
# Scrape Creators API
# ============================================================================


def get_transcript_from_scrape_creators(youtube_url: str) -> str:
    """Get transcript from Scrape Creators API."""
    api_key = os.getenv("SCRAPECREATORS_API_KEY")
    if not api_key:
        raise ValueError("SCRAPECREATORS_API_KEY environment variable is required")

    print(f"🔗 Fetching transcript from Scrape Creators API...")
    start_time = time.time()

    url = f"https://api.scrapecreators.com/v1/youtube/video?url={youtube_url}&get_transcript=true"
    headers = {"x-api-key": api_key}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    result = response.json()
    transcript = result.get("transcript_only_text", "")

    elapsed = time.time() - start_time
    print(f"✅ Transcript fetched in {elapsed:.2f}s ({len(transcript)} characters)")

    if not transcript:
        raise ValueError("No transcript found in API response")

    return transcript


# ============================================================================
# Main
# ============================================================================


def main():
    """Main test function."""
    test_url = "https://youtu.be/pmdiKAE_GLs"
    model = Config.ANALYSIS_MODEL

    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]

    print("=" * 80)
    print("TEST: Scrape Creators API + OpenRouter with LangGraph Verification")
    print("=" * 80)
    print(f"📹 Video URL: {test_url}")
    print(f"🤖 Model: {model}\n")

    try:
        # Step 1: Get transcript
        transcript = get_transcript_from_scrape_creators(test_url)

        # Step 2: Analyze with LangGraph workflow
        print("\n" + "=" * 80)
        print("ANALYSIS & VERIFICATION WORKFLOW")
        print("=" * 80)
        graph = create_compiled_graph()

        start_time = time.time()
        result: dict = graph.invoke(GraphInput(transcript=transcript, analysis_model=model))
        result: GraphOutput = GraphOutput.model_validate(result)
        elapsed = time.time() - start_time

        # Step 3: Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\n⏱️  Total time: {elapsed:.2f}s")
        print(f"🔄 Iterations: {result.iteration_count}")
        print(f"📈 Final quality score: {result.quality.percentage_score}%")
        print(f"\n📝 Summary:")
        print(f"{result.analysis.summary}")
        print(f"\n🎯 Takeaways ({len(result.analysis.takeaways)}):")
        for i, takeaway in enumerate(result.analysis.takeaways, 1):
            print(f"  {i}. {takeaway}")
        print(f"\n📊 Key Facts ({len(result.analysis.key_facts)}):")
        for i, fact in enumerate(result.analysis.key_facts, 1):
            print(f"  {i}. {fact}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
