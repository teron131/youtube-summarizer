"""
This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using LangChain with LangGraph self-checking workflow.
"""

import os
from typing import Optional

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
MAX_ITERATIONS = 3


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

    transcript_or_url: str


class WorkflowOutput(BaseModel):
    """Output from the summarization workflow."""

    analysis: Analysis
    quality: Quality
    iteration_count: int


class WorkflowState(BaseModel):
    """Flattened state for the summarization workflow."""

    # Input
    transcript_or_url: str

    # Analysis results
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None

    # Control fields
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


def langchain_llm(model: str) -> BaseChatModel:
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


def langchain_or_gemini(state: WorkflowState) -> str:
    """Determine whether to use Gemini SDK or LangChain based on input type."""
    return "gemini_analysis" if is_youtube_url(state.transcript_or_url) else "langchain_analysis"


def gemini_analysis_node(state: WorkflowState) -> dict:
    """Generate initial analysis using Gemini SDK for YouTube URLs."""
    print(f"🔗 Processing YouTube URL with Gemini SDK: {state.transcript_or_url}")

    # Use Gemini SDK directly for YouTube URLs
    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="models/gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction=ANALYSIS_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    result = response.parsed
    print(f"📊 Gemini SDK analysis completed")

    return {
        "analysis": result,
        "iteration_count": state.iteration_count + 1,
    }


def langchain_analysis_node(state: WorkflowState) -> dict:
    """Generate initial analysis using LangChain for transcript text."""
    print(f"📝 Sending transcript text to LangChain LLM: {len(state.transcript_or_url)} characters")
    print(f"📝 Text preview: {state.transcript_or_url[:200]}...")

    # Use LangChain for transcript text
    llm = langchain_llm(ANALYSIS_MODEL)
    structured_llm = llm.with_structured_output(Analysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANALYSIS_PROMPT),
            ("human", "{content}"),
        ]
    )

    chain = prompt | structured_llm
    result = chain.invoke({"content": state.transcript_or_url})
    print(f"📊 LangChain analysis completed")

    return {
        "analysis": result,
        "iteration_count": state.iteration_count + 1,
    }


def gemini_quality_node(state: WorkflowState) -> dict:
    """Check quality using Gemini SDK."""
    print("🔍 Performing quality check with Gemini SDK...")

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Convert analysis to text for evaluation
    analysis_text = f"""
Title: {state.analysis.title}
Summary: {state.analysis.summary}
Takeaways: {', '.join(state.analysis.takeaways)}
Key Facts: {', '.join(state.analysis.key_facts)}
Keywords: {', '.join(state.analysis.keywords)}
Chapters: {len(state.analysis.chapters)} chapters
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
                types.Part(text=analysis_text),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction=QUALITY_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=Quality,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )

    quality = response.parsed
    print(f"📈 Gemini quality score: {quality.score}/100")

    if quality.issues:
        print(f"⚠️  Issues found: {', '.join(quality.issues)}")

    return {
        "quality": quality,
        "is_complete": quality.score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS,
    }


def langchain_quality_node(state: WorkflowState) -> dict:
    """Check the quality of the generated analysis using LangChain."""
    print("🔍 Performing quality check with LangChain...")

    llm = langchain_llm(QUALITY_MODEL)
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

    print(f"📈 LangChain quality score: {quality.score}/100")

    if quality.issues:
        print(f"⚠️  Issues found: {', '.join(quality.issues)}")

    return {
        "quality": quality,
        "is_complete": quality.score >= MIN_QUALITY_SCORE or state.iteration_count >= MAX_ITERATIONS,
    }


def gemini_refinement_node(state: WorkflowState) -> dict:
    """Refine analysis using Gemini SDK."""
    print("🔧 Refining analysis with Gemini SDK...")

    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    # chr(10) is '\n'
    improvement_prompt = f"""# Improve this video analysis based on the following feedback:

## Issues to Address:
{chr(10).join(f"- {issue}" for issue in state.quality.issues)}

## Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in state.quality.suggestions)}

## Original Analysis:

### Title
{state.analysis.title}

### Summary
{state.analysis.summary}

### Key Takeaways
{chr(10).join(f"- {takeaway}" for takeaway in state.analysis.takeaways)}

### Key Facts
{chr(10).join(f"- {fact}" for fact in state.analysis.key_facts)}

Please provide an improved version that addresses these issues.
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-pro",
        contents=types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
                types.Part(text=improvement_prompt),
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction="Improve the analysis based on the feedback while maintaining the same structure and format.",
            temperature=0,
            response_mime_type="application/json",
            response_schema=Analysis,
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )

    result = response.parsed
    print(f"✨ Gemini SDK analysis refined (iteration {state.iteration_count + 1})")
    return {"analysis": result, "iteration_count": state.iteration_count + 1}


def langchain_refinement_node(state: WorkflowState) -> dict:
    """Refine the analysis based on quality feedback using LangChain."""
    print("🔧 Refining analysis with LangChain...")

    llm = langchain_llm(ANALYSIS_MODEL)
    structured_llm = llm.with_structured_output(Analysis)

    # chr(10) is '\n'
    improvement_prompt = f"""# Improve this video analysis based on the following feedback:

## Issues to Address:
{chr(10).join(f"- {issue}" for issue in state.quality.issues)}

## Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in state.quality.suggestions)}

## Original Analysis:

### Title
{state.analysis.title}

### Summary
{state.analysis.summary}

### Key Takeaways
{chr(10).join(f"- {takeaway}" for takeaway in state.analysis.takeaways)}

### Key Facts
{chr(10).join(f"- {fact}" for fact in state.analysis.key_facts)}

Please provide an improved version that addresses these issues.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Improve the analysis based on the feedback while maintaining the same structure and format."),
            ("human", "{improvement_prompt}"),
        ]
    )

    chain = prompt | structured_llm
    result: Analysis = chain.invoke({"improvement_prompt": improvement_prompt})
    print(f"✨ LangChain analysis refined (iteration {state.iteration_count + 1})")

    return {"analysis": result, "iteration_count": state.iteration_count + 1}


def should_continue_gemini(state: WorkflowState) -> str:
    """Determine next step in Gemini workflow."""
    if state.is_complete:
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        return "gemini_refinement"
    else:
        return END


def should_continue_langchain(state: WorkflowState) -> str:
    """Determine next step in LangChain workflow."""
    if state.is_complete:
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < MAX_ITERATIONS:
        return "langchain_refinement"
    else:
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(WorkflowState, input=WorkflowInput, output=WorkflowOutput)

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
            graph.get_graph().draw_mermaid_png(
                output_file_path="graph.png",
            )
        except Exception as e:
            print(f"Error generating graph image: {e}")

    return graph


def summarize_video(transcript_or_url: str) -> Analysis:
    """
    Summarize the text using LangChain with LangGraph self-checking workflow.
    """
    # Create and run the workflow
    graph = create_compiled_graph()

    # Run the workflow
    result: WorkflowOutput = graph.invoke(WorkflowInput(transcript_or_url=transcript_or_url))
    result = WorkflowOutput.model_validate(result)  # LangGraph returns a dictionary instead of Pydantic model

    print(f"🎯 Final quality score: {result.quality.score}/100 (after {result.iteration_count} iterations)")
    return result.analysis
