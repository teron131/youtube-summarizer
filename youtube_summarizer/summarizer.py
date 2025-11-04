"""
This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using LangChain with LangGraph self-checking workflow.
"""

import os
from typing import Generator, Literal, Optional, Union

from dotenv import load_dotenv
from google.genai import Client, types
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .utils import is_youtube_url, schema_to_string

load_dotenv()


# Global configuration
ANALYSIS_MODEL = "google/gemini-2.5-pro"
QUALITY_MODEL = "google/gemini-2.5-flash"
MIN_QUALITY_SCORE = 90
MAX_ITERATIONS = 2

# Translation configuration
ENABLE_TRANSLATION = False
TARGET_LANGUAGE = "zh-TW"  # ISO language code (en, es, fr, de, etc.)


class Chapter(BaseModel):
    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")


class Analysis(BaseModel):
    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience")
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned")
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="The exact keywords in the analysis worthy of highlighting")
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")


class Rate(BaseModel):
    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect (Fail=poor, Refine=adequate, Pass=excellent)")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    grammar: Rate = Field(description="Rate for grammar: No typos, grammatical mistakes, appropriate wordings")
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")
    meta_language_avoidance: Rate = Field(description="Rate for meta-language avoidance: No phrases like 'This chapter introduces', 'This section covers', etc.")
    useful_keywords: Rate = Field(description="Rate for keywords: The keywords are useful for highlighting the analysis")
    correct_language: Rate = Field(description="Rate for language: Match the original language of the transcript or user requested")

    # Computed properties
    @property
    def total_score(self) -> int:
        """Calculate total score based on all quality aspects."""
        score_map = {"Fail": 0, "Refine": 1, "Pass": 2}
        aspects = [self.completeness, self.structure, self.grammar, self.no_garbage, self.useful_keywords, self.correct_language]
        return sum(score_map[aspect.rate] for aspect in aspects)

    @property
    def max_possible_score(self) -> int:
        """Calculate maximum possible score (all aspects = Pass)."""
        return len([self.completeness, self.structure, self.grammar, self.no_garbage, self.useful_keywords, self.correct_language]) * 2

    @property
    def percentage_score(self) -> int:
        """Calculate percentage score out of 100."""
        return int((self.total_score / self.max_possible_score) * 100)

    @property
    def is_acceptable(self) -> bool:
        """Whether the analysis meets quality standards (score >= 90%)."""
        return self.percentage_score >= MIN_QUALITY_SCORE


class GraphInput(BaseModel):
    """Input for the summarization workflow."""

    transcript_or_url: str

    # Model selection
    analysis_model: str = Field(default=ANALYSIS_MODEL)
    quality_model: str = Field(default=QUALITY_MODEL)

    # Translation options
    target_language: Optional[str] = Field(default=None)


class GraphOutput(BaseModel):
    """Output from the summarization workflow."""

    analysis: Analysis
    quality: Quality
    iteration_count: int


class GraphState(BaseModel):
    """Flattened state for the summarization workflow."""

    # Input
    transcript_or_url: str

    # Model selection
    analysis_model: str = Field(default=ANALYSIS_MODEL)
    quality_model: str = Field(default=QUALITY_MODEL)

    # Translation options
    target_language: Optional[str] = Field(default=None)

    # Analysis results
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None

    # Control fields
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


# Prompt templates
def get_analysis_prompt(state: GraphState) -> str:
    """Generate streamlined analysis prompt with essential quality guidelines."""
    schema = schema_to_string(Analysis)
    base_prompt = f"""Create a comprehensive analysis that strictly follows the transcript content.

OUTPUT SCHEMA:
{schema}

CORE REQUIREMENTS:
- ACCURACY: Every claim must be directly supported by the transcript
- LENGTH: Summary (150-400 words), Chapters (80-200 words each), Takeaways (3-8), Key Facts (3-6)
- TONE: Write in objective, article-like style (avoid "This video...", "The speaker...")
- AVOID META-DESCRIPTIVE LANGUAGE: Do not use phrases like "This chapter introduces", "This section covers", "This analysis explores", etc. Write direct, factual content only

CONTENT FILTERING:
- Remove all promotional content (speaker intros, calls-to-action, self-promotion)
- Keep only educational content about the strategy
- Correct obvious typos naturally

CHAPTER REQUIREMENTS:
"""

    # Generate chapters based on content structure and topic transitions
    base_prompt += """Create 4-8 thematic chapters based on content structure and topic transitions."""

    base_prompt += """

QUALITY CHECKS:
- Content matches transcript exactly (no external additions)
- All promotional content removed (intros, calls-to-action, self-promotion)
- Typos corrected naturally, meaning preserved
- Length balanced: substantial but not overwhelming
- Keywords highly relevant and searchable"""

    if state.target_language:
        base_prompt += f"""

TRANSLATION:
- Translate to {state.target_language} with natural fluency
- Preserve technical terms and proper names
- Maintain same detail level and structure"""
    return base_prompt


def get_quality_prompt(state: GraphState) -> str:
    """Generate streamlined quality evaluation prompt."""
    base_prompt = """Evaluate the analysis on 10 aspects. Rate each "Fail", "Refine", or "Pass" with a specific reason.

ASPECTS:
1. TRANSCRIPT ACCURACY: Content directly supported by transcript (no external additions)
2. CONTENT LENGTH: Balanced length following guidelines (not too short/long)
3. COMPLETENESS: Entire content properly analyzed
4. STRUCTURE: Follows required schema perfectly with takeaways/key_facts as simple arrays
5. GRAMMAR: No typos/mistakes (semicolon usage in lists acceptable)
6. WRITING STYLE: Objective, article-like tone (no video references or meta-descriptive phrases like "This chapter introduces")
7. PROMOTIONAL REMOVAL: All promotional content completely removed
8. META-LANGUAGE AVOIDANCE: No phrases like "This chapter introduces", "This section covers", "This analysis explores", etc.
9. ARRAY VALIDITY: Takeaways/key_facts are valid arrays with appropriate content
10. KEYWORDS: Highly relevant and useful"""

    if state.target_language:
        base_prompt += f"""
11. TRANSLATION QUALITY: Content is properly translated to {state.target_language} with natural fluency and maintained quality"""
    else:
        base_prompt += """
11. LANGUAGE CONSISTENCY: Content matches original language"""

    base_prompt += """

LENGTH GUIDELINES:
- Title: 2-15 words
- Summary: 150-400 words
- Chapters: 80-200 words each
- Takeaways: 3-8 items
- Key Facts: 3-6 items
- Keywords: Exactly 3

QUALITY STANDARDS:
- Transcript-based content only
- Natural chapter flow
- Professional article-like tone

Provide specific rates and reasons for each aspect."""

    # Provide explicit output schema to guide structured responses
    base_prompt += f"""

OUTPUT SCHEMA:
{schema_to_string(Quality)}"""
    return base_prompt


def get_improvement_prompt(state: GraphState) -> str:
    """Generate streamlined improvement prompt."""
    base_prompt = """Improve the analysis based on quality feedback while maintaining transcript accuracy.

IMPROVEMENT PRIORITIES:
1. TRANSCRIPT ACCURACY: All content must be transcript-supported
2. PROMOTIONAL REMOVAL: Remove all intros, calls-to-action, self-promotion
3. LENGTH BALANCE: Follow guidelines (Summary: 150-400 words, Chapters: 80-200 each)
4. WRITING STYLE: Use objective, article-like tone (avoid "This video...", "The speaker...")
5. AVOID META-DESCRIPTIVE LANGUAGE: Remove phrases like "This chapter introduces", "This section covers", etc. Write direct, factual content only
6. TYPO CORRECTION: Fix obvious typos naturally
7. ARRAY FORMATTING: Return takeaways/key_facts as simple string arrays

CONTENT TARGETS:
- Title: 2-15 words
- Summary: 150-400 words
- Chapters: 80-200 words each
- Takeaways: Simple array with 3-8 strings
- Key Facts: Simple array with 3-6 strings
- Keywords: Exactly 3"""

    if state.target_language:
        base_prompt += f"""

TRANSLATION IMPROVEMENT REQUIREMENTS:
- Maintain all content in {state.target_language}
- Preserve translation quality while fixing identified issues
- Keep technical terms and proper names appropriate for the target language
- Maintain natural fluency and cultural relevance"""

    # Provide explicit output schema to guide structured responses
    base_prompt += f"""

OUTPUT SCHEMA:
{schema_to_string(Analysis)}"""

    return base_prompt


def langchain_llm(model: str) -> BaseChatModel:
    """Create LangChain LLM instance based on model format."""
    if "/" in model:
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
    else:
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
        )


class ContextProcessor:
    """Base class for analysis processing with common functionality."""

    @staticmethod
    def create_improvement_prompt(analysis: Analysis, quality: Quality) -> str:
        """Create improvement prompt from analysis and quality feedback."""
        return f"""# Improve this video analysis based on the following feedback:

## Analysis:
{repr(analysis)}

## Quality Assessment:
{repr(quality)}

Please provide an improved version that addresses the specific issues identified above to improve the overall quality score."""


# Workflow node functions
def langchain_or_gemini(state: GraphState) -> str:
    """Determine whether to use Gemini SDK or LangChain based on input type."""
    return "gemini_analysis" if is_youtube_url(state.transcript_or_url) else "langchain_analysis"


def langchain_analysis_node(state: GraphState) -> dict[str, Union[Analysis, int]]:
    """Super node: generate or refine analysis using LangChain.

    - If there is no prior quality feedback, generate a fresh analysis.
    - If quality feedback exists, refine using the feedback and original context.
    """
    # Skip LangChain if no transcript is available
    if state.transcript_or_url is None:
        print("🎯 No transcript available - skipping LangChain analysis")
        # Create a minimal analysis indicating no transcript
        result = Analysis(title="No Transcript Available", summary="This video does not have a transcript available for analysis.", takeaways=["No transcript available"], key_facts=["No transcript available"], chapters=[], keywords=["no-transcript"])
        result.target_language = state.target_language if state.target_language else None
        return {"analysis": result, "iteration_count": state.iteration_count + 1}

    print(f"📝 Using model: {state.analysis_model}")
    llm = langchain_llm(state.analysis_model)
    structured_llm = llm.with_structured_output(Analysis)

    # Refinement path when previous quality feedback exists
    if state.quality is not None and state.analysis is not None:
        print("🔧 Super node mode: refining analysis with LangChain based on quality feedback...")
        improvement_prompt = ContextProcessor.create_improvement_prompt(state.analysis, state.quality)
        improvement_system_prompt = get_improvement_prompt(state)

        # Include original transcript context to anchor refinements
        transcript_context = f"Original Transcript:\n{state.transcript_or_url}"
        full_improvement_prompt = f"{transcript_context}\n\n{improvement_prompt}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", improvement_system_prompt),
                ("human", "{improvement_prompt}"),
            ]
        )
        chain = prompt | structured_llm
        try:
            result: Analysis = chain.invoke({"improvement_prompt": full_improvement_prompt})
            result.target_language = state.target_language if state.target_language else None
            print("✨ LangChain super node refined analysis")
            return {"analysis": result, "iteration_count": state.iteration_count + 1}
        except Exception as e:
            print(f"❌ LangChain refinement failed: {str(e)}")
            # Return the original analysis if refinement fails
            return {"analysis": state.analysis, "iteration_count": state.iteration_count + 1}

    # Generation path (no previous feedback)
    print(f"📝 Super node mode: generating initial analysis. Transcript length: {len(state.transcript_or_url)}")
    print(f"📝 Text preview: {state.transcript_or_url[:200]}...")
    if state.target_language:
        print(f"🌐 Translation enabled: {state.target_language}")

    content = state.transcript_or_url
    analysis_prompt = get_analysis_prompt(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", analysis_prompt),
            ("human", "{content}"),
        ]
    )
    chain = prompt | structured_llm
    try:
        result: Analysis = chain.invoke({"content": content})
        result.target_language = state.target_language if state.target_language else None
        print("📊 LangChain analysis completed")
        return {"analysis": result, "iteration_count": state.iteration_count + 1}
    except Exception as e:
        print(f"❌ LangChain analysis failed: {str(e)}")
        # Create a minimal fallback analysis
        fallback_analysis = Analysis(title="Analysis Generation Failed", summary=f"Unable to generate structured analysis due to: {str(e)[:100]}. The transcript was processed but could not be analyzed.", takeaways=["Analysis failed due to technical issues"], key_facts=["Processing encountered errors"], chapters=[], keywords=["error"])
        fallback_analysis.target_language = state.target_language if state.target_language else None
        return {"analysis": fallback_analysis, "iteration_count": state.iteration_count + 1}


def langchain_quality_node(state: GraphState) -> dict[str, Union[Quality, bool]]:
    """Check the quality of the generated analysis using LangChain."""
    print("🔍 Performing quality check with LangChain...")
    print(f"🔍 Using model: {state.quality_model}")

    llm = langchain_llm(state.quality_model)
    structured_llm = llm.with_structured_output(Quality)

    quality_prompt = get_quality_prompt(state)
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

        # is_acceptable is now computed automatically by the Quality model
        print(f"📈 LangChain quality breakdown:")
        print(f"Completeness: {quality.completeness.rate} - {quality.completeness.reason}")
        print(f"Structure: {quality.structure.rate} - {quality.structure.reason}")
        print(f"Grammar: {quality.grammar.rate} - {quality.grammar.reason}")
        print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
        print(f"Meta-Language Avoidance: {quality.meta_language_avoidance.rate} - {quality.meta_language_avoidance.reason}")
        print(f"Useful Keywords: {quality.useful_keywords.rate} - {quality.useful_keywords.reason}")
        print(f"Correct Language: {quality.correct_language.rate} - {quality.correct_language.reason}")
        print(f"Total Score: {quality.total_score}/{quality.max_possible_score} ({quality.percentage_score}%)")

        if not quality.is_acceptable:
            print(f"⚠️  Quality below threshold ({MIN_QUALITY_SCORE}%), refinement needed")

        return {
            "quality": quality,
            "is_complete": quality.percentage_score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS,
        }
    except Exception as e:
        print(f"❌ LangChain quality check failed: {str(e)}")
        # Create a minimal fallback quality assessment
        fallback_quality = Quality(completeness=Rate(rate="Fail", reason=f"Quality check failed: {str(e)[:50]}"), structure=Rate(rate="Fail", reason="Unable to assess structure"), grammar=Rate(rate="Fail", reason="Unable to assess grammar"), no_garbage=Rate(rate="Fail", reason="Unable to assess content quality"), meta_language_avoidance=Rate(rate="Fail", reason="Unable to assess meta-language"), useful_keywords=Rate(rate="Fail", reason="Unable to assess keywords"), correct_language=Rate(rate="Fail", reason="Unable to assess language correctness"))
        print(f"📈 Fallback quality score: {fallback_quality.percentage_score}%")

        return {
            "quality": fallback_quality,
            "is_complete": state.iteration_count >= MAX_ITERATIONS,  # Complete if max iterations reached
        }


def gemini_analysis_node(state: GraphState) -> dict[str, Union[Analysis, int]]:
    """Super node: generate or refine analysis using Gemini SDK for URLs."""
    print(f"🔗 Using model: {state.analysis_model}")
    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Refinement path when quality feedback exists
    if state.quality is not None and state.analysis is not None:
        print("🔧 Super node mode: refining analysis with Gemini SDK based on quality feedback...")
        improvement_prompt = ContextProcessor.create_improvement_prompt(state.analysis, state.quality)
        improvement_system_prompt = get_improvement_prompt(state)

        # Prepare content parts with full context
        content_parts = [types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)), types.Part(text=improvement_prompt)]

        try:
            response = client.models.generate_content(
                model=state.analysis_model.split("/")[1] if "google/" in state.analysis_model else "gemini-2.5-pro",
                contents=types.Content(parts=content_parts),
                config=types.GenerateContentConfig(
                    system_instruction=improvement_system_prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=Analysis,
                    thinking_config=types.ThinkingConfig(thinking_budget=2048),
                ),
            )
            result: Analysis = response.parsed
            result.target_language = state.target_language if state.target_language else None
            print("✨ Gemini SDK super node refined analysis")
            return {"analysis": result, "iteration_count": state.iteration_count + 1}
        except Exception as e:
            print(f"❌ Gemini SDK refinement failed: {str(e)}")
            # Return the original analysis if refinement fails
            return {"analysis": state.analysis, "iteration_count": state.iteration_count + 1}

    # Generation path
    print(f"🔗 Super node mode: generating initial analysis. URL: {state.transcript_or_url}")
    analysis_prompt = get_analysis_prompt(state)
    content_parts = [types.Part(file_data=types.FileData(file_uri=state.transcript_or_url))]

    try:
        response = client.models.generate_content(
            model=state.analysis_model.split("/")[1] if "google/" in state.analysis_model else "gemini-2.5-pro",
            contents=types.Content(parts=content_parts),
            config=types.GenerateContentConfig(
                system_instruction=analysis_prompt,
                temperature=0,
                response_mime_type="application/json",
                response_schema=Analysis,
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
            ),
        )
        result: Analysis = response.parsed
        result.target_language = state.target_language if state.target_language else None
        print("📊 Gemini SDK analysis completed")
        return {"analysis": result, "iteration_count": state.iteration_count + 1}
    except Exception as e:
        print(f"❌ Gemini SDK analysis failed: {str(e)}")
        # Create a minimal fallback analysis for Gemini
        fallback_analysis = Analysis(title="Gemini Analysis Failed", summary=f"Unable to generate analysis using Gemini SDK due to: {str(e)[:100]}. The content was processed but analysis failed.", takeaways=["Gemini SDK encountered an error"], key_facts=["Technical issues with Gemini API"], chapters=[], keywords=["gemini-error"])
        fallback_analysis.target_language = state.target_language if state.target_language else None
        return {"analysis": fallback_analysis, "iteration_count": state.iteration_count + 1}


def gemini_quality_node(state: GraphState) -> dict[str, Union[Quality, bool]]:
    """Check quality using Gemini SDK."""
    print("🔍 Performing quality check with Gemini SDK...")
    print(f"🔍 Using model: {state.quality_model}")

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    quality_prompt = get_quality_prompt(state)
    analysis_text = repr(state.analysis)

    try:
        response = client.models.generate_content(
            model=state.quality_model.split("/")[1] if "google/" in state.quality_model else "gemini-2.5-flash",
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
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
            ),
        )

        quality: Quality = response.parsed

        # is_acceptable is now computed automatically by the Quality model
        print(f"📈 Gemini quality breakdown:")
        print(f"Completeness: {quality.completeness.rate} - {quality.completeness.reason}")
        print(f"Structure: {quality.structure.rate} - {quality.structure.reason}")
        print(f"Grammar: {quality.grammar.rate} - {quality.grammar.reason}")
        print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
        print(f"Meta-Language Avoidance: {quality.meta_language_avoidance.rate} - {quality.meta_language_avoidance.reason}")
        print(f"Useful Keywords: {quality.useful_keywords.rate} - {quality.useful_keywords.reason}")
        print(f"Correct Language: {quality.correct_language.rate} - {quality.correct_language.reason}")
        print(f"Total Score: {quality.total_score}/{quality.max_possible_score} ({quality.percentage_score}%)")

        if not quality.is_acceptable:
            print(f"⚠️  Quality below threshold ({MIN_QUALITY_SCORE}%), refinement needed")

        return {
            "quality": quality,
            "is_complete": quality.percentage_score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS,
        }
    except Exception as e:
        print(f"❌ Gemini SDK quality check failed: {str(e)}")
        # Create a minimal fallback quality assessment for Gemini
        fallback_quality = Quality(completeness=Rate(rate="Fail", reason=f"Gemini quality check failed: {str(e)[:50]}"), structure=Rate(rate="Fail", reason="Unable to assess structure"), grammar=Rate(rate="Fail", reason="Unable to assess grammar"), no_garbage=Rate(rate="Fail", reason="Unable to assess content quality"), meta_language_avoidance=Rate(rate="Fail", reason="Unable to assess meta-language"), useful_keywords=Rate(rate="Fail", reason="Unable to assess keywords"), correct_language=Rate(rate="Fail", reason="Unable to assess language correctness"))
        print(f"📈 Fallback Gemini quality score: {fallback_quality.percentage_score}%")

        return {
            "quality": fallback_quality,
            "is_complete": state.iteration_count >= MAX_ITERATIONS,  # Complete if max iterations reached
        }


# Conditional routing functions
def should_continue_langchain(state: GraphState) -> str:
    """Determine next step in LangChain workflow."""
    if state.is_complete:
        print(f"🔄 LangChain workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 LangChain quality {state.quality.percentage_score}% below threshold {MIN_QUALITY_SCORE}%, re-entering analysis super node (iteration {state.iteration_count + 1})")
        return "langchain_analysis"
    else:
        print(f"🔄 LangChain workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def should_continue_gemini(state: GraphState) -> str:
    """Determine next step in Gemini workflow."""
    if state.is_complete:
        print(f"🔄 Gemini workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 Gemini quality {state.quality.percentage_score}% below threshold {MIN_QUALITY_SCORE}%, re-entering analysis super node (iteration {state.iteration_count + 1})")
        return "gemini_analysis"
    else:
        print(f"🔄 Gemini workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("langchain_analysis", langchain_analysis_node)
    builder.add_node("langchain_quality", langchain_quality_node)
    builder.add_node("gemini_analysis", gemini_analysis_node)
    builder.add_node("gemini_quality", gemini_quality_node)

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


def summarize_video(transcript_or_url: str) -> Analysis:
    """Summarize the text using LangChain with LangGraph self-checking workflow."""
    graph = create_compiled_graph()

    # LangGraph returns a dictionary instead of Pydantic model
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

    # LangGraph returns a dictionary instead of Pydantic model
    for chunk in graph.stream(
        GraphInput(transcript_or_url=transcript_or_url),
        stream_mode="values",
    ):
        yield GraphState.model_validate(chunk)
