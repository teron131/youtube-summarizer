"""
This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using LangChain with LangGraph self-checking workflow.
"""

import os
from typing import Generator, Literal, Optional

from dotenv import load_dotenv
from google.genai import Client, types
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from youtube_summarizer.utils import is_youtube_url

load_dotenv()


# Global configuration
ANALYSIS_MODEL = "google/gemini-2.5-pro"
QUALITY_MODEL = "google/gemini-2.5-flash"
MIN_QUALITY_SCORE = 90
MAX_ITERATIONS = 2

# Translation configuration
ENABLE_TRANSLATION = False
TARGET_LANGUAGE = "en"  # ISO language code (en, es, fr, de, etc.)


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
    keywords: list[str] = Field(description="Keywords or topics mentioned in the video, max 3", max_length=3)
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")


class Rate(BaseModel):
    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect (Fail=poor, Refine=adequate, Pass=excellent)")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    grammar: Rate = Field(description="Rate for grammar: No typos, grammatical mistakes, appropriate wordings")
    timestamp: Rate = Field(description="Rate for timestamp: The timestamps added are in correct format")
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")
    correct_language: Rate = Field(description="Rate for language: Match the original language of the transcript or user requested")

    # Computed properties
    @property
    def total_score(self) -> int:
        """Calculate total score based on all quality aspects."""
        score_map = {"Fail": 0, "Refine": 1, "Pass": 2}
        aspects = [self.completeness, self.structure, self.grammar, self.timestamp, self.no_garbage, self.correct_language]
        return sum(score_map[aspect.rate] for aspect in aspects)

    @property
    def max_possible_score(self) -> int:
        """Calculate maximum possible score (all aspects = Pass)."""
        return len([self.completeness, self.structure, self.grammar, self.timestamp, self.no_garbage, self.correct_language]) * 2

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
    enable_translation: bool = Field(default=ENABLE_TRANSLATION)
    target_language: str = Field(default=TARGET_LANGUAGE)


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
    enable_translation: bool = Field(default=ENABLE_TRANSLATION)
    target_language: str = Field(default=TARGET_LANGUAGE)

    # Analysis results
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None

    # Control fields
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


# Prompt templates
def get_analysis_prompt(state: GraphState) -> str:
    """Generate analysis prompt with optional translation instructions."""
    base_prompt = """Analyze the video/transcript according to the schema.
The transcript provides starting timestamps for each sentence.
Add the timestamps [TIMESTAMP] at the end of the takeaways and key facts if available.
Consider the chapters (headers) if given but not necessary.
Ignore the promotional and meaningless content."""

    if state.enable_translation:
        language_name = state.target_language
        base_prompt += f"\n\nIMPORTANT: Translate the entire analysis to {language_name} ({state.target_language}) while maintaining structure and natural fluency."
    return base_prompt


def get_quality_prompt(state: GraphState) -> str:
    """Generate quality evaluation prompt with optional translation context."""
    base_prompt = """Evaluate the quality of this video analysis on these 6 aspects. For each aspect, give a rate of "Fail", "Refine", or "Pass" and provide a single reason.

Scoring Guide:
- "Fail" = Poor/Incomplete/Inaccurate (needs significant improvement)
- "Refine" = Adequate/Partially complete/Somewhat accurate (needs some improvement)
- "Pass" = Excellent/Complete/Accurate (meets quality standards)

Aspects to evaluate:
1. Completeness: The entire video has been considered
2. Structure: The result is in desired structures
3. Grammar: No typos, grammatical mistakes, appropriate wordings
4. Timestamp: The timestamps added are in correct format
5. No Garbage: The promotional and meaningless content are removed"""

    if state.enable_translation:
        language_name = state.target_language
        base_prompt += f"""
6. Correct Language: Content is properly translated to {language_name} and maintains quality"""
    else:
        base_prompt += """
6. Correct Language: Match the original language of the video or user requested"""

    base_prompt += """

Provide rates and reasons for each aspect. The total score will be calculated automatically."""
    return base_prompt


def get_improvement_prompt(state: GraphState) -> str:
    """Generate improvement prompt with optional translation context."""
    base_prompt = """Improve the analysis based on the feedback while maintaining the same structure and format."""

    if state.enable_translation:
        language_name = state.target_language
        base_prompt += f"""
IMPORTANT: Ensure all improvements maintain proper translation to {language_name} ({state.target_language}).
- Keep all content translated to {language_name}
- Maintain translation quality while addressing the specific feedback issues"""

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

## Quality Assessment (Total: {quality.total_score}/{quality.max_possible_score} - {quality.percentage_score}%):

### Completeness: {quality.completeness.rate}
{quality.completeness.reason}

### Structure: {quality.structure.rate}
{quality.structure.reason}

### Grammar: {quality.grammar.rate}
{quality.grammar.reason}

### Timestamp: {quality.timestamp.rate}
{quality.timestamp.reason}

### No Garbage: {quality.no_garbage.rate}
{quality.no_garbage.reason}

### Correct Language: {quality.correct_language.rate}
{quality.correct_language.reason}

## Original Analysis:

### Title
{analysis.title}

### Summary
{analysis.summary}

### Key Takeaways
{chr(10).join(f"- {takeaway}" for takeaway in analysis.takeaways)}

### Key Facts
{chr(10).join(f"- {fact}" for fact in analysis.key_facts)}

Please provide an improved version that addresses the specific issues identified above to improve the overall quality score."""

    @staticmethod
    def analysis_to_text(analysis: Analysis) -> str:
        """Convert analysis to text format for quality evaluation."""
        return f"""
Title: {analysis.title}
Summary: {analysis.summary}
Takeaways: {', '.join(analysis.takeaways)}
Key Facts: {', '.join(analysis.key_facts)}
Keywords: {', '.join(analysis.keywords)}
Chapters: {len(analysis.chapters)} chapters
"""


# Workflow node functions
def langchain_or_gemini(state: GraphState) -> str:
    """Determine whether to use Gemini SDK or LangChain based on input type."""
    return "gemini_analysis" if is_youtube_url(state.transcript_or_url) else "langchain_analysis"


def langchain_analysis_node(state: GraphState) -> dict:
    """Generate initial analysis using LangChain for transcript text."""
    print(f"📝 Sending transcript text to LangChain LLM: {len(state.transcript_or_url)} characters")
    print(f"📝 Text preview: {state.transcript_or_url[:200]}...")
    print(f"📝 Using model: {state.analysis_model}")
    if state.enable_translation:
        print(f"🌐 Translation enabled: {state.target_language}")

    llm = langchain_llm(state.analysis_model)
    structured_llm = llm.with_structured_output(Analysis)

    analysis_prompt = get_analysis_prompt(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", analysis_prompt),
            ("human", "{content}"),
        ]
    )

    chain = prompt | structured_llm
    result: Analysis = chain.invoke({"content": state.transcript_or_url})

    result.target_language = state.target_language if state.enable_translation else None

    print(f"📊 LangChain analysis completed")
    return {
        "analysis": result,
        "iteration_count": state.iteration_count + 1,
    }


def langchain_quality_node(state: GraphState) -> dict:
    """Check the quality of the generated analysis using LangChain."""
    print("🔍 Performing quality check with LangChain...")
    print(f"🔍 Using model: {state.quality_model}")

    llm = langchain_llm(state.quality_model)
    structured_llm = llm.with_structured_output(Quality)

    analysis_text = ContextProcessor.analysis_to_text(state.analysis)
    quality_prompt = get_quality_prompt(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", quality_prompt),
            ("human", "{analysis_text}"),
        ]
    )

    chain = prompt | structured_llm
    quality: Quality = chain.invoke({"analysis_text": analysis_text})

    # is_acceptable is now computed automatically by the Quality model
    print(f"📈 LangChain quality breakdown:")
    print(f"Completeness: {quality.completeness.rate} - {quality.completeness.reason}")
    print(f"Structure: {quality.structure.rate} - {quality.structure.reason}")
    print(f"Grammar: {quality.grammar.rate} - {quality.grammar.reason}")
    print(f"Timestamp: {quality.timestamp.rate} - {quality.timestamp.reason}")
    print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
    print(f"Correct Language: {quality.correct_language.rate} - {quality.correct_language.reason}")
    print(f"Total Score: {quality.total_score}/{quality.max_possible_score} ({quality.percentage_score}%)")

    if not quality.is_acceptable:
        print(f"⚠️  Quality below threshold ({MIN_QUALITY_SCORE}%), refinement needed")

    return {
        "quality": quality,
        "is_complete": quality.percentage_score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS,
    }


def langchain_refinement_node(state: GraphState) -> dict:
    """Refine the analysis based on quality feedback using LangChain."""
    print("🔧 Refining analysis with LangChain...")
    print(f"🔧 Using model: {state.analysis_model}")

    llm = langchain_llm(state.analysis_model)
    structured_llm = llm.with_structured_output(Analysis)

    improvement_prompt = ContextProcessor.create_improvement_prompt(state.analysis, state.quality)
    improvement_system_prompt = get_improvement_prompt(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", improvement_system_prompt),
            ("human", "{improvement_prompt}"),
        ]
    )

    chain = prompt | structured_llm
    result: Analysis = chain.invoke({"improvement_prompt": improvement_prompt})

    # Update translation metadata
    result.target_language = state.target_language if state.enable_translation else None

    print(f"✨ LangChain analysis refined")
    return {
        "analysis": result,
        "iteration_count": state.iteration_count + 1,
    }


def gemini_analysis_node(state: GraphState) -> dict:
    """Generate initial analysis using Gemini SDK for YouTube URLs."""
    print(f"🔗 Processing YouTube URL with Gemini SDK: {state.transcript_or_url}")
    print(f"🔗 Using model: {state.analysis_model}")
    if state.enable_translation:
        print(f"🌐 Translation enabled: {state.target_language}")

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))
    analysis_prompt = get_analysis_prompt(state)

    response = client.models.generate_content(
        model=state.analysis_model.split("/")[1] if "google/" in state.analysis_model else "gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction=analysis_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    result: Analysis = response.parsed

    # Update translation metadata
    result.target_language = state.target_language if state.enable_translation else None

    print(f"📊 Gemini SDK analysis completed")
    return {
        "analysis": result,
        "iteration_count": state.iteration_count + 1,
    }


def gemini_quality_node(state: GraphState) -> dict:
    """Check quality using Gemini SDK."""
    print("🔍 Performing quality check with Gemini SDK...")
    print(f"🔍 Using model: {state.quality_model}")

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))
    analysis_text = ContextProcessor.analysis_to_text(state.analysis)
    quality_prompt = get_quality_prompt(state)

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
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )

    quality: Quality = response.parsed

    # is_acceptable is now computed automatically by the Quality model
    print(f"📈 Gemini quality breakdown:")
    print(f"Completeness: {quality.completeness.rate} - {quality.completeness.reason}")
    print(f"Structure: {quality.structure.rate} - {quality.structure.reason}")
    print(f"Grammar: {quality.grammar.rate} - {quality.grammar.reason}")
    print(f"Timestamp: {quality.timestamp.rate} - {quality.timestamp.reason}")
    print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
    print(f"Correct Language: {quality.correct_language.rate} - {quality.correct_language.reason}")
    print(f"Total Score: {quality.total_score}/{quality.max_possible_score} ({quality.percentage_score}%)")

    if not quality.is_acceptable:
        print(f"⚠️  Quality below threshold ({MIN_QUALITY_SCORE}%), refinement needed")

    return {
        "quality": quality,
        "is_complete": quality.percentage_score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS,
    }


def gemini_refinement_node(state: GraphState) -> dict:
    """Refine analysis using Gemini SDK."""
    print("🔧 Refining analysis with Gemini SDK...")
    print(f"🔧 Using model: {state.analysis_model}")

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))
    improvement_prompt = ContextProcessor.create_improvement_prompt(state.analysis, state.quality)
    improvement_system_prompt = get_improvement_prompt(state)

    response = client.models.generate_content(
        model=state.analysis_model.split("/")[1] if "google/" in state.analysis_model else "gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
                types.Part(text=improvement_prompt),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction=improvement_system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    result: Analysis = response.parsed

    # Update translation metadata
    result.target_language = state.target_language if state.enable_translation else None

    print(f"✨ Gemini SDK analysis refined")
    return {
        "analysis": result,
        "iteration_count": state.iteration_count + 1,
    }


# Conditional routing functions
def should_continue_langchain(state: GraphState) -> str:
    """Determine next step in LangChain workflow."""
    if state.is_complete:
        print(f"🔄 LangChain workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 LangChain quality {state.quality.percentage_score}% below threshold {MIN_QUALITY_SCORE}%, continuing to refinement (iteration {state.iteration_count + 1})")
        return "langchain_refinement"
    else:
        print(f"🔄 LangChain workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def should_continue_gemini(state: GraphState) -> str:
    """Determine next step in Gemini workflow."""
    if state.is_complete:
        print(f"🔄 Gemini workflow complete (is_complete=True)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        print(f"🔄 Gemini quality {state.quality.percentage_score}% below threshold {MIN_QUALITY_SCORE}%, continuing to refinement (iteration {state.iteration_count + 1})")
        return "gemini_refinement"
    else:
        print(f"🔄 Gemini workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("gemini_analysis", gemini_analysis_node)
    builder.add_node("langchain_analysis", langchain_analysis_node)
    builder.add_node("gemini_quality", gemini_quality_node)
    builder.add_node("langchain_quality", langchain_quality_node)
    builder.add_node("gemini_refinement", gemini_refinement_node)
    builder.add_node("langchain_refinement", langchain_refinement_node)

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
    builder.add_edge("gemini_analysis", "gemini_quality")
    builder.add_edge("langchain_analysis", "langchain_quality")

    # Add conditional edges from quality nodes
    builder.add_conditional_edges(
        "gemini_quality",
        should_continue_gemini,
        {
            "gemini_refinement": "gemini_refinement",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "langchain_quality",
        should_continue_langchain,
        {
            "langchain_refinement": "langchain_refinement",
            END: END,
        },
    )

    # Add edges from refinement nodes back to quality nodes
    builder.add_edge("gemini_refinement", "gemini_quality")
    builder.add_edge("langchain_refinement", "langchain_quality")

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
