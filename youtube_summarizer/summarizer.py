"""
This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using LangChain with LangGraph self-checking workflow.
"""

import os
from typing import Any, Generator, Literal, Optional, Union

from dotenv import load_dotenv
from google.genai import Client, types
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .utils import is_youtube_url, schema_to_string

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Centralized configuration for the summarization workflow."""

    # Model configuration
    ANALYSIS_MODEL = "x-ai/grok-4-fast"
    QUALITY_MODEL = "x-ai/grok-4-fast"

    # Quality thresholds
    MIN_QUALITY_SCORE = 90
    MAX_ITERATIONS = 2

    # Translation configuration
    ENABLE_TRANSLATION = False
    TARGET_LANGUAGE = "zh-TW"  # ISO language code (en, es, fr, de, etc.)


# ============================================================================
# Data Models
# ============================================================================


class Chapter(BaseModel):
    """Represents a single chapter in the analysis."""

    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")


class Analysis(BaseModel):
    """Complete analysis of video content."""

    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(
        description="Key insights and actionable takeaways for the audience",
        min_items=3,
        max_items=8,
    )
    key_facts: list[str] = Field(
        description="Important facts, statistics, or data points mentioned",
        min_items=3,
        max_items=6,
    )
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(
        description="The most relevant keywords in the analysis worthy of highlighting",
        min_items=3,
        max_items=3,
    )
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect (Fail=poor, Refine=adequate, Pass=excellent)")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment of the analysis."""

    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    grammar: Rate = Field(description="Rate for grammar: No typos, grammatical mistakes, appropriate wordings")
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")
    meta_language_avoidance: Rate = Field(description="Rate for meta-language avoidance: No phrases like 'This chapter introduces', 'This section covers', etc.")
    useful_keywords: Rate = Field(description="Rate for keywords: The keywords are useful for highlighting the analysis")
    correct_language: Rate = Field(description="Rate for language: Match the original language of the transcript or user requested")

    @property
    def all_aspects(self) -> list[Rate]:
        """Return all quality aspects as a list."""
        return [
            self.completeness,
            self.structure,
            self.grammar,
            self.no_garbage,
            self.useful_keywords,
            self.correct_language,
        ]

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

    transcript_or_url: str
    analysis_model: str = Field(default=Config.ANALYSIS_MODEL)
    quality_model: str = Field(default=Config.QUALITY_MODEL)
    target_language: Optional[str] = Field(default=None)


class GraphOutput(BaseModel):
    """Output from the summarization workflow."""

    analysis: Analysis
    quality: Quality
    iteration_count: int


class GraphState(BaseModel):
    """Flattened state for the summarization workflow."""

    transcript_or_url: str
    analysis_model: str = Field(default=Config.ANALYSIS_MODEL)
    quality_model: str = Field(default=Config.QUALITY_MODEL)
    target_language: Optional[str] = Field(default=None)
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


# ============================================================================
# Prompt Builder
# ============================================================================


class PromptBuilder:
    """Centralized prompt builder that extracts requirements from Pydantic model Field descriptions."""

    # Field display name mappings
    ANALYSIS_FIELD_MAPPING = {
        "title": "Title",
        "summary": "Summary",
        "takeaways": "Takeaways",
        "key_facts": "Key Facts",
        "chapters": "Chapters",
        "keywords": "Keywords",
    }

    QUALITY_ASPECT_MAPPING = {
        "completeness": "COMPLETENESS",
        "structure": "STRUCTURE",
        "grammar": "GRAMMAR",
        "no_garbage": "PROMOTIONAL REMOVAL",
        "meta_language_avoidance": "META-LANGUAGE AVOIDANCE",
        "useful_keywords": "KEYWORDS",
        "correct_language": "LANGUAGE CONSISTENCY",
    }

    @staticmethod
    def _extract_field_info(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
        """Extract field descriptions and constraints from a Pydantic model."""
        schema = model.model_json_schema()
        fields_info = {}

        for field_name, field_info in schema.get("properties", {}).items():
            fields_info[field_name] = {
                "description": field_info.get("description", ""),
                "type": field_info.get("type"),
                "min_items": field_info.get("minItems"),
                "max_items": field_info.get("maxItems"),
                "required": field_name in schema.get("required", []),
            }

        return fields_info

    @staticmethod
    def _build_field_requirements(model: type[BaseModel], field_mapping: dict[str, str] | None = None) -> str:
        """Build field requirements section from model Field descriptions."""
        fields_info = PromptBuilder._extract_field_info(model)
        lines = []

        for field_name, info in fields_info.items():
            if not info["description"]:
                continue

            display_name = (field_mapping or {}).get(field_name, field_name.replace("_", " ").title())
            requirement = f"- {display_name}: {info['description']}"

            # Add constraints
            min_items = info.get("min_items")
            max_items = info.get("max_items")
            if min_items is not None and max_items is not None:
                if min_items == max_items:
                    requirement += f" (Exactly {min_items} items)"
                else:
                    requirement += f" ({min_items}-{max_items} items)"
            elif min_items is not None:
                requirement += f" (At least {min_items} items)"
            elif max_items is not None:
                requirement += f" (At most {max_items} items)"

            lines.append(requirement)

        return "\n".join(lines)

    @staticmethod
    def build_analysis_prompt(state: GraphState) -> str:
        """Build analysis prompt from Analysis model Field descriptions."""
        schema = schema_to_string(Analysis)
        field_requirements = PromptBuilder._build_field_requirements(Analysis, PromptBuilder.ANALYSIS_FIELD_MAPPING)

        prompt_parts = [
            "Create a comprehensive analysis that strictly follows the transcript content.",
            "",
            "OUTPUT SCHEMA:",
            schema,
            "",
            "FIELD REQUIREMENTS:",
            field_requirements,
            "",
            "CORE REQUIREMENTS:",
            "- ACCURACY: Every claim must be directly supported by the transcript",
            "- TONE: Write in objective, article-like style (avoid 'This video...', 'The speaker...')",
            "- AVOID META-DESCRIPTIVE LANGUAGE: Do not use phrases like 'This chapter introduces', 'This section covers', 'This analysis explores', etc. Write direct, factual content only",
            "",
            "CONTENT FILTERING:",
            "- Remove all promotional content (speaker intros, calls-to-action, self-promotion)",
            "- Keep only educational content about the strategy",
            "- Correct obvious typos naturally",
            "",
            "CHAPTER REQUIREMENTS:",
            "Create 4-8 thematic chapters based on content structure and topic transitions.",
            "",
            "QUALITY CHECKS:",
            "- Content matches transcript exactly (no external additions)",
            "- All promotional content removed (intros, calls-to-action, self-promotion)",
            "- Typos corrected naturally, meaning preserved",
            "- Length balanced: substantial but not overwhelming",
            "- Keywords highly relevant and searchable",
        ]

        if state.target_language:
            prompt_parts.extend(
                [
                    "",
                    "TRANSLATION:",
                    f"- Translate to {state.target_language} with natural fluency",
                    "- Preserve technical terms and proper names",
                    "- Maintain same detail level and structure",
                ]
            )

        return "\n".join(prompt_parts)

    @staticmethod
    def build_quality_prompt(state: GraphState) -> str:
        """Build quality evaluation prompt from Quality model Field descriptions."""
        fields_info = PromptBuilder._extract_field_info(Quality)
        schema = schema_to_string(Quality)

        # Build aspects list from Quality model fields
        aspects_lines = []
        for idx, (field_name, info) in enumerate(fields_info.items(), 1):
            if field_name == "correct_language" and state.target_language:
                aspect_name = "TRANSLATION QUALITY"
                description = f"Content is properly translated to {state.target_language} with natural fluency and maintained quality"
            else:
                aspect_name = PromptBuilder.QUALITY_ASPECT_MAPPING.get(field_name, field_name.upper().replace("_", " "))
                desc = info["description"]
                description = desc.split(":", 1)[1].strip() if ":" in desc else desc

            aspects_lines.append(f"{idx}. {aspect_name}: {description}")

        # Build length guidelines from Analysis model
        analysis_fields = PromptBuilder._extract_field_info(Analysis)
        length_lines = []
        for field_name, info in analysis_fields.items():
            min_items = info.get("min_items")
            max_items = info.get("max_items")

            if field_name == "title":
                length_lines.append("- Title: 2-15 words")
            elif field_name == "summary":
                length_lines.append("- Summary: 150-400 words")
            elif field_name == "chapters":
                length_lines.append("- Chapters: 80-200 words each")
            elif field_name == "takeaways" and min_items is not None and max_items is not None:
                length_lines.append(f"- Takeaways: {min_items}-{max_items} items")
            elif field_name == "key_facts" and min_items is not None and max_items is not None:
                length_lines.append(f"- Key Facts: {min_items}-{max_items} items")
            elif field_name == "keywords" and min_items is not None and max_items is not None:
                if min_items == max_items:
                    length_lines.append(f"- Keywords: Exactly {min_items}")
                else:
                    length_lines.append(f"- Keywords: {min_items}-{max_items} items")

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
            "- Natural chapter flow",
            "- Professional article-like tone",
            "",
            "Provide specific rates and reasons for each aspect.",
            "",
            "OUTPUT SCHEMA:",
            schema,
        ]

        return "\n".join(prompt_parts)

    @staticmethod
    def build_improvement_prompt(state: GraphState) -> str:
        """Build improvement prompt from Analysis model Field descriptions."""
        schema = schema_to_string(Analysis)
        field_requirements = PromptBuilder._build_field_requirements(Analysis, PromptBuilder.ANALYSIS_FIELD_MAPPING)

        prompt_parts = [
            "Improve the analysis based on quality feedback while maintaining transcript accuracy.",
            "",
            "IMPROVEMENT PRIORITIES:",
            "1. TRANSCRIPT ACCURACY: All content must be transcript-supported",
            "2. PROMOTIONAL REMOVAL: Remove all intros, calls-to-action, self-promotion",
            "3. WRITING STYLE: Use objective, article-like tone (avoid 'This video...', 'The speaker...')",
            "4. AVOID META-DESCRIPTIVE LANGUAGE: Remove phrases like 'This chapter introduces', 'This section covers', etc. Write direct, factual content only",
            "5. TYPO CORRECTION: Fix obvious typos naturally",
            "6. ARRAY FORMATTING: Return takeaways/key_facts as simple string arrays",
            "",
            "CONTENT TARGETS:",
            field_requirements,
        ]

        if state.target_language:
            prompt_parts.extend(
                [
                    "",
                    "TRANSLATION IMPROVEMENT REQUIREMENTS:",
                    f"- Maintain all content in {state.target_language}",
                    "- Preserve translation quality while fixing identified issues",
                    "- Keep technical terms and proper names appropriate for the target language",
                    "- Maintain natural fluency and cultural relevance",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "OUTPUT SCHEMA:",
                schema,
            ]
        )

        return "\n".join(prompt_parts)


# ============================================================================
# Model Clients
# ============================================================================


class ModelClientFactory:
    """Factory for creating model clients (LangChain or Gemini SDK)."""

    @staticmethod
    def create_langchain_llm(model: str) -> BaseChatModel:
        """Create LangChain LLM instance based on model format."""
        if "/" in model:
            # OpenRouter format (e.g., "google/gemini-2.5-flash")
            return init_chat_model(
                model=model,
                model_provider="openai",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
            )
        else:
            # Native Gemini format (e.g., "gemini-2.5-flash")
            return init_chat_model(
                model=model,
                model_provider="google_genai",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.0,
            )

    @staticmethod
    def create_gemini_client() -> Client:
        """Create Gemini SDK client."""
        return Client(api_key=os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def extract_gemini_model_name(model: str) -> str:
        """Extract Gemini model name from OpenRouter format or return default."""
        if "google/" in model:
            return model.split("/")[1]
        return Config.GEMINI_DEFAULT_MODEL


# ============================================================================
# Utilities
# ============================================================================


class AnalysisUtils:
    """Utility functions for analysis operations."""

    @staticmethod
    def create_fallback_analysis(error_msg: str, error_type: str, state: GraphState) -> Analysis:
        """Create a fallback analysis when generation fails."""
        analysis = Analysis(
            title=f"{error_type} Analysis Failed",
            summary=f"Unable to generate analysis due to: {error_msg[:100]}. The content was processed but analysis failed.",
            takeaways=[f"{error_type} encountered an error"],
            key_facts=["Technical issues with analysis"],
            chapters=[],
            keywords=[error_type.lower().replace(" ", "-")],
        )
        if state.target_language:
            analysis.target_language = state.target_language
        return analysis

    @staticmethod
    def create_no_transcript_analysis(state: GraphState) -> Analysis:
        """Create analysis indicating no transcript is available."""
        analysis = Analysis(
            title="No Transcript Available",
            summary="This video does not have a transcript available for analysis.",
            takeaways=["No transcript available"],
            key_facts=["No transcript available"],
            chapters=[],
            keywords=["no-transcript"],
        )
        if state.target_language:
            analysis.target_language = state.target_language
        return analysis

    @staticmethod
    def set_target_language(analysis: Analysis, target_language: Optional[str]) -> None:
        """Set target language on analysis if provided."""
        if target_language:
            analysis.target_language = target_language


class QualityUtils:
    """Utility functions for quality operations."""

    @staticmethod
    def create_fallback_quality(error_msg: str) -> Quality:
        """Create a fallback quality assessment when quality check fails."""
        return Quality(
            completeness=Rate(rate="Fail", reason=f"Quality check failed: {error_msg[:50]}"),
            structure=Rate(rate="Fail", reason="Unable to assess structure"),
            grammar=Rate(rate="Fail", reason="Unable to assess grammar"),
            no_garbage=Rate(rate="Fail", reason="Unable to assess content quality"),
            meta_language_avoidance=Rate(rate="Fail", reason="Unable to assess meta-language"),
            useful_keywords=Rate(rate="Fail", reason="Unable to assess keywords"),
            correct_language=Rate(rate="Fail", reason="Unable to assess language correctness"),
        )

    @staticmethod
    def print_quality_breakdown(quality: Quality, provider: str = "") -> None:
        """Print quality breakdown with all aspects."""
        prefix = f"📈 {provider} quality breakdown:" if provider else "📈 Quality breakdown:"
        print(prefix)
        print(f"Completeness: {quality.completeness.rate} - {quality.completeness.reason}")
        print(f"Structure: {quality.structure.rate} - {quality.structure.reason}")
        print(f"Grammar: {quality.grammar.rate} - {quality.grammar.reason}")
        print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
        print(f"Meta-Language Avoidance: {quality.meta_language_avoidance.rate} - {quality.meta_language_avoidance.reason}")
        print(f"Useful Keywords: {quality.useful_keywords.rate} - {quality.useful_keywords.reason}")
        print(f"Correct Language: {quality.correct_language.rate} - {quality.correct_language.reason}")
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
# Analysis Nodes (Shared Base Logic)
# ============================================================================


class AnalysisNodeBase:
    """Base class for analysis nodes with shared logic."""

    @staticmethod
    def _is_refinement_mode(state: GraphState) -> bool:
        """Check if we're in refinement mode (have previous quality feedback)."""
        return state.quality is not None and state.analysis is not None

    @staticmethod
    def _prepare_refinement_inputs(state: GraphState) -> tuple[str, str]:
        """Prepare inputs for refinement mode."""
        improvement_context = PromptUtils.create_improvement_context(state.analysis, state.quality)
        improvement_system_prompt = PromptBuilder.build_improvement_prompt(state)
        transcript_context = f"Original Transcript:\n{state.transcript_or_url}"
        full_improvement_prompt = f"{transcript_context}\n\n{improvement_context}"
        return improvement_system_prompt, full_improvement_prompt

    @staticmethod
    def _handle_result(result: Analysis, state: GraphState, success_msg: str) -> dict[str, Union[Analysis, int]]:
        """Handle successful result by setting target language and returning."""
        AnalysisUtils.set_target_language(result, state.target_language)
        print(success_msg)
        return {"analysis": result, "iteration_count": state.iteration_count + 1}

    @staticmethod
    def _handle_error(error: Exception, state: GraphState, error_type: str) -> dict[str, Union[Analysis, int]]:
        """Handle error by creating fallback or returning original."""
        print(f"❌ {error_type} failed: {str(error)}")
        if state.analysis:
            # Return original analysis if refinement fails
            return {"analysis": state.analysis, "iteration_count": state.iteration_count + 1}
        # Create fallback for generation failures
        fallback = AnalysisUtils.create_fallback_analysis(str(error), error_type, state)
        return {"analysis": fallback, "iteration_count": state.iteration_count + 1}


class LangChainAnalysisNode(AnalysisNodeBase):
    """LangChain-based analysis node."""

    @staticmethod
    def execute(state: GraphState) -> dict[str, Union[Analysis, int]]:
        """Execute LangChain analysis node."""
        if state.transcript_or_url is None:
            print("🎯 No transcript available - skipping LangChain analysis")
            result = AnalysisUtils.create_no_transcript_analysis(state)
            return {"analysis": result, "iteration_count": state.iteration_count + 1}

        print(f"📝 Using model: {state.analysis_model}")
        llm = ModelClientFactory.create_langchain_llm(state.analysis_model)
        structured_llm = llm.with_structured_output(Analysis)

        if AnalysisNodeBase._is_refinement_mode(state):
            print("🔧 Super node mode: refining analysis with LangChain based on quality feedback...")
            system_prompt, improvement_prompt = AnalysisNodeBase._prepare_refinement_inputs(state)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{improvement_prompt}"),
                ]
            )
            chain = prompt | structured_llm

            try:
                result: Analysis = chain.invoke({"improvement_prompt": improvement_prompt})
                return AnalysisNodeBase._handle_result(result, state, "✨ LangChain super node refined analysis")
            except Exception as e:
                return AnalysisNodeBase._handle_error(e, state, "LangChain refinement")

        # Generation path
        print(f"📝 Super node mode: generating initial analysis. Transcript length: {len(state.transcript_or_url)}")
        print(f"📝 Text preview: {state.transcript_or_url[:200]}...")
        if state.target_language:
            print(f"🌐 Translation enabled: {state.target_language}")

        analysis_prompt = PromptBuilder.build_analysis_prompt(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", analysis_prompt),
                ("human", "{content}"),
            ]
        )
        chain = prompt | structured_llm

        try:
            result: Analysis = chain.invoke({"content": state.transcript_or_url})
            return AnalysisNodeBase._handle_result(result, state, "📊 LangChain analysis completed")
        except Exception as e:
            return AnalysisNodeBase._handle_error(e, state, "LangChain analysis")


class GeminiAnalysisNode(AnalysisNodeBase):
    """Gemini SDK-based analysis node."""

    @staticmethod
    def execute(state: GraphState) -> dict[str, Union[Analysis, int]]:
        """Execute Gemini analysis node."""
        print(f"🔗 Using model: {state.analysis_model}")
        client = ModelClientFactory.create_gemini_client()
        model_name = ModelClientFactory.extract_gemini_model_name(state.analysis_model)

        if AnalysisNodeBase._is_refinement_mode(state):
            print("🔧 Super node mode: refining analysis with Gemini SDK based on quality feedback...")
            system_prompt, improvement_prompt = AnalysisNodeBase._prepare_refinement_inputs(state)

            content_parts = [
                types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
                types.Part(text=improvement_prompt),
            ]

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=content_parts),
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=Analysis,
                        thinking_config=types.ThinkingConfig(thinking_budget=Config.GEMINI_THINKING_BUDGET),
                    ),
                )
                result: Analysis = response.parsed
                return AnalysisNodeBase._handle_result(result, state, "✨ Gemini SDK super node refined analysis")
            except Exception as e:
                return AnalysisNodeBase._handle_error(e, state, "Gemini SDK refinement")

        # Generation path
        print(f"🔗 Super node mode: generating initial analysis. URL: {state.transcript_or_url}")
        analysis_prompt = PromptBuilder.build_analysis_prompt(state)
        content_parts = [types.Part(file_data=types.FileData(file_uri=state.transcript_or_url))]

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(parts=content_parts),
                config=types.GenerateContentConfig(
                    system_instruction=analysis_prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=Analysis,
                    thinking_config=types.ThinkingConfig(thinking_budget=Config.GEMINI_THINKING_BUDGET),
                ),
            )
            result: Analysis = response.parsed
            return AnalysisNodeBase._handle_result(result, state, "📊 Gemini SDK analysis completed")
        except Exception as e:
            return AnalysisNodeBase._handle_error(e, state, "Gemini SDK analysis")


# ============================================================================
# Quality Nodes (Shared Base Logic)
# ============================================================================


class QualityNodeBase:
    """Base class for quality nodes with shared logic."""

    @staticmethod
    def _calculate_completion(quality: Quality, state: GraphState) -> bool:
        """Calculate if workflow should complete."""
        return quality.percentage_score >= Config.MIN_QUALITY_SCORE or state.iteration_count >= Config.MAX_ITERATIONS


class LangChainQualityNode(QualityNodeBase):
    """LangChain-based quality node."""

    @staticmethod
    def execute(state: GraphState) -> dict[str, Union[Quality, bool]]:
        """Execute LangChain quality check."""
        print("🔍 Performing quality check with LangChain...")
        print(f"🔍 Using model: {state.quality_model}")

        llm = ModelClientFactory.create_langchain_llm(state.quality_model)
        structured_llm = llm.with_structured_output(Quality)

        quality_prompt = PromptBuilder.build_quality_prompt(state)
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
            QualityUtils.print_quality_breakdown(quality, "LangChain")

            return {
                "quality": quality,
                "is_complete": QualityNodeBase._calculate_completion(quality, state),
            }
        except Exception as e:
            print(f"❌ LangChain quality check failed: {str(e)}")
            fallback_quality = QualityUtils.create_fallback_quality(str(e))
            print(f"📈 Fallback quality score: {fallback_quality.percentage_score}%")

            return {
                "quality": fallback_quality,
                "is_complete": state.iteration_count >= Config.MAX_ITERATIONS,
            }


class GeminiQualityNode(QualityNodeBase):
    """Gemini SDK-based quality node."""

    @staticmethod
    def execute(state: GraphState) -> dict[str, Union[Quality, bool]]:
        """Execute Gemini quality check."""
        print("🔍 Performing quality check with Gemini SDK...")
        print(f"🔍 Using model: {state.quality_model}")

        client = ModelClientFactory.create_gemini_client()
        model_name = ModelClientFactory.extract_gemini_model_name(state.quality_model)
        if model_name == Config.GEMINI_DEFAULT_MODEL:
            model_name = Config.GEMINI_QUALITY_MODEL

        quality_prompt = PromptBuilder.build_quality_prompt(state)
        analysis_text = repr(state.analysis)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(
                    parts=[
                        types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
                        types.Part(text=analysis_text),
                    ]
                ),
                config=types.GenerateContentConfig(
                    system_instruction=quality_prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=Quality,
                    thinking_config=types.ThinkingConfig(thinking_budget=Config.GEMINI_THINKING_BUDGET),
                ),
            )

            quality: Quality = response.parsed
            QualityUtils.print_quality_breakdown(quality, "Gemini")

            return {
                "quality": quality,
                "is_complete": QualityNodeBase._calculate_completion(quality, state),
            }
        except Exception as e:
            print(f"❌ Gemini SDK quality check failed: {str(e)}")
            fallback_quality = QualityUtils.create_fallback_quality(str(e))
            print(f"📈 Fallback Gemini quality score: {fallback_quality.percentage_score}%")

            return {
                "quality": fallback_quality,
                "is_complete": state.iteration_count >= Config.MAX_ITERATIONS,
            }


# ============================================================================
# Graph Workflow
# ============================================================================


def langchain_or_gemini(state: GraphState) -> str:
    """Determine whether to use Gemini SDK or LangChain based on input type."""
    return "gemini_analysis" if is_youtube_url(state.transcript_or_url) else "langchain_analysis"


def should_continue_langchain(state: GraphState) -> str:
    """Determine next step in LangChain workflow."""
    if state.is_complete:
        print(f"🔄 LangChain workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < Config.MAX_ITERATIONS:
        print(f"🔄 LangChain quality {state.quality.percentage_score}% below threshold {Config.MIN_QUALITY_SCORE}%, re-entering analysis super node (iteration {state.iteration_count + 1})")
        return "langchain_analysis"
    else:
        print(f"🔄 LangChain workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def should_continue_gemini(state: GraphState) -> str:
    """Determine next step in Gemini workflow."""
    if state.is_complete:
        print(f"🔄 Gemini workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < Config.MAX_ITERATIONS:
        print(f"🔄 Gemini quality {state.quality.percentage_score}% below threshold {Config.MIN_QUALITY_SCORE}%, re-entering analysis super node (iteration {state.iteration_count + 1})")
        return "gemini_analysis"
    else:
        print(f"🔄 Gemini workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("langchain_analysis", LangChainAnalysisNode.execute)
    builder.add_node("langchain_quality", LangChainQualityNode.execute)
    builder.add_node("gemini_analysis", GeminiAnalysisNode.execute)
    builder.add_node("gemini_quality", GeminiQualityNode.execute)

    # Add conditional routing from START
    builder.add_conditional_edges(
        START,
        langchain_or_gemini,
        {
            "langchain_analysis": "langchain_analysis",
            "gemini_analysis": "gemini_analysis",
        },
    )

    # Add edges from analysis nodes to quality nodes
    builder.add_edge("langchain_analysis", "langchain_quality")
    builder.add_edge("gemini_analysis", "gemini_quality")

    # Add conditional edges from quality nodes
    builder.add_conditional_edges(
        "langchain_quality",
        should_continue_langchain,
        {
            "langchain_analysis": "langchain_analysis",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "gemini_quality",
        should_continue_gemini,
        {
            "gemini_analysis": "gemini_analysis",
            END: END,
        },
    )

    return builder


def create_compiled_graph():
    """Create and compile the summarization graph."""
    graph = create_summarization_graph().compile()

    if "graph.png" not in os.listdir():
        try:
            graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        except Exception as e:
            print(f"Error generating graph image: {e}")

    return graph


# ============================================================================
# Public API
# ============================================================================


def summarize_video(transcript_or_url: str) -> Analysis:
    """Summarize the text using LangChain with LangGraph self-checking workflow."""
    graph = create_compiled_graph()

    result: dict = graph.invoke(GraphInput(transcript_or_url=transcript_or_url))
    result: GraphOutput = GraphOutput.model_validate(result)

    print(f"🎯 Final quality score: {result.quality.percentage_score}% (after {result.iteration_count} iterations)")
    return result.analysis


def stream_summarize_video(transcript_or_url: str) -> Generator[GraphState, None, None]:
    """Stream the summarization process with progress updates using LangGraph's stream_mode='values'.

    This allows for both getting adhoc progress status updates and the final result.

    The final chunk will contain the complete graph state with the final analysis.
    """
    graph = create_compiled_graph()

    for chunk in graph.stream(
        GraphInput(transcript_or_url=transcript_or_url),
        stream_mode="values",
    ):
        yield GraphState.model_validate(chunk)
