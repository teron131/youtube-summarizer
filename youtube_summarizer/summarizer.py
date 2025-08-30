"""
This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using LangChain with LangGraph self-checking workflow.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from youtube_summarizer.utils import is_youtube_url

load_dotenv()


class Chapter(BaseModel):
    header: str = Field(description="A descriptive title for the chapter")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")


class Analysis(BaseModel):
    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience")
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned")
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="Keywords or topics mentioned in the video, max 3", max_length=3)


class Quality(BaseModel):
    score: int = Field(description="Quality score from 0-100", ge=0, le=100)
    issues: list[str] = Field(description="List of identified quality issues")
    suggestions: list[str] = Field(description="Suggestions for improvement")
    is_acceptable: bool = Field(description="Whether the analysis meets quality standards")


ANALYSIS_PROMPT = """Analyze the video/transcript according to the schema.
The transcript provides starting timestamps for each sentence.
Add the timestamps [TIMESTAMP] at the end of the takeaways and key facts if available.
Consider the chapters (headers) if given but not necessary.
Ignore the promotional and meaningless content.
Follow the original language."""

QUALITY_PROMPT = """Evaluate the quality of this video analysis on these aspects:
1. Completeness (are all key points covered?)
2. Accuracy (does it faithfully represent the content?)
3. Structure (is it well-organized?)
4. Clarity (is it easy to understand?)
5. Actionability (are takeaways useful?)
Provide an overall score from 0-100 and specific feedback."""


class WorkflowInput(BaseModel):
    """Input for the summarization workflow."""

    content: str


class WorkflowOutput(BaseModel):
    """Output from the summarization workflow."""

    analysis: Analysis
    quality: Quality
    iteration_count: int


class WorkflowState(BaseModel):
    """Flattened state for the summarization workflow."""

    # Input
    content: str

    # Analysis results
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None

    # Control fields
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


# Constants
ANALYSIS_MODEL = "google/gemini-2.5-pro"
QUALITY_MODEL = "google/gemini-2.5-flash"
MIN_QUALITY_SCORE = 80
MAX_ITERATIONS = 2


def get_llm(model: str) -> BaseChatModel:
    """Create LLM instance based on model format."""
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


def initial_analysis_node(state: WorkflowState) -> dict:
    """Generate initial analysis of the content."""
    if is_youtube_url(state.content):
        print(f"🔗 Processing YouTube URL: {state.content}")
        content = f"YouTube video URL: {state.content}"
    else:
        print(f"📝 Sending transcript text to LLM: {len(state.content)} characters")
        print(f"📝 Text preview: {state.content[:200]}...")
        content = state.content

    llm = get_llm(ANALYSIS_MODEL)
    structured_llm = llm.with_structured_output(Analysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYSIS_PROMPT),
            ("human", "{content}"),
        ]
    )

    chain = prompt | structured_llm
    result = chain.invoke({"content": content})

    print(f"📊 Initial analysis completed")
    return {"analysis": result, "iteration_count": state.iteration_count + 1}


def quality_node(state: WorkflowState) -> dict:
    """Check the quality of the generated analysis."""
    print("🔍 Performing quality check...")

    llm = get_llm(QUALITY_MODEL)
    structured_llm = llm.with_structured_output(Quality)

    # Convert analysis to text for evaluation
    analysis_text = f"""
Title: {state.analysis.title}
Summary: {state.analysis.summary}
Takeaways: {', '.join(state.analysis.takeaways)}
Key Facts: {', '.join(state.analysis.key_facts)}
Keywords: {', '.join(state.analysis.keywords)}
Chapters: {len(state.analysis.chapters)} chapters
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUALITY_PROMPT),
            ("human", "{analysis_text}"),
        ]
    )

    chain = prompt | structured_llm
    quality: Quality = chain.invoke({"analysis_text": analysis_text})

    print(f"📈 Quality score: {quality.score}/100")

    if quality.issues:
        print(f"⚠️  Issues found: {', '.join(quality.issues)}")

    return {"quality": quality, "is_complete": quality.score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS}


def refinement_node(state: WorkflowState) -> dict:
    """Refine the analysis based on quality feedback."""
    print("🔧 Refining analysis based on feedback...")

    llm = get_llm(ANALYSIS_MODEL)
    structured_llm = llm.with_structured_output(Analysis)

    improvement_prompt = f"""
Improve this video analysis based on the following feedback:

Issues: {', '.join(state.quality.issues)}
Suggestions: {', '.join(state.quality.suggestions)}

Original Analysis:
Title: {state.analysis.title}
Summary: {state.analysis.summary}
Takeaways: {', '.join(state.analysis.takeaways)}
Key Facts: {', '.join(state.analysis.key_facts)}

Please provide an improved version that addresses these issues.
"""

    prompt = ChatPromptTemplate.from_messages([("system", "Improve the analysis based on the feedback while maintaining the same structure and format."), ("human", "{improvement_prompt}")])

    chain = prompt | structured_llm
    result: Analysis = chain.invoke({"improvement_prompt": improvement_prompt})

    print(f"✨ Analysis refined (iteration {state.iteration_count + 1})")
    return {"analysis": result, "iteration_count": state.iteration_count + 1}


def should_continue(state: WorkflowState) -> str:
    """Determine next step in the workflow."""
    if state.is_complete:
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        return "refinement"
    else:
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph."""
    builder = StateGraph(WorkflowState, input=WorkflowInput, output=WorkflowOutput)

    # Add nodes
    builder.add_node("initial_analysis", initial_analysis_node)
    builder.add_node("quality", quality_node)
    builder.add_node("refinement", refinement_node)

    # Add edges
    builder.add_edge(START, "initial_analysis")
    builder.add_edge("initial_analysis", "quality")
    builder.add_edge("refinement", "quality")

    # Conditional edges
    builder.add_conditional_edges(
        "quality",
        should_continue,
        {
            "refinement": "refinement",
            END: END,
        },
    )

    return builder


def create_compiled_graph():
    """Create and compile the summarization graph."""
    return create_summarization_graph().compile()


def summarize_video(url_or_transcript: str) -> Analysis:
    """
    Summarize the text using LangChain with LangGraph self-checking workflow.
    """
    # Create and run the workflow
    graph = create_compiled_graph()

    # Run the workflow
    result = graph.invoke(WorkflowInput(content=url_or_transcript))

    # Debug: print the actual result structure
    print(f"🔍 Debug - Result keys: {list(result.keys())}")
    print(f"🔍 Debug - Result: {result}")

    # Extract values from the result state (result is a dict with the final state)
    analysis: Analysis = result.get("analysis")
    quality: Quality = result.get("quality")
    iteration_count: int = result.get("iteration_count", 0)

    print(f"🎯 Final quality score: {quality.score}/100 (after {iteration_count} iterations)")

    return analysis
