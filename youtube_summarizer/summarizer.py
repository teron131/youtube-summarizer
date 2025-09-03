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
    keywords: list[str] = Field(description="The exact keywords in the analysis worthy of highlighting, max 3", max_length=3)
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")


class Rate(BaseModel):
    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect (Fail=poor, Refine=adequate, Pass=excellent)")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    grammar: Rate = Field(description="Rate for grammar: No typos, grammatical mistakes, appropriate wordings")
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")
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
    chapters: list[dict] = Field(default_factory=list)  # YouTube video chapters

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
    chapters: list[dict] = Field(default_factory=list)  # YouTube video chapters

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
    base_prompt = """Create a comprehensive analysis that strictly follows the transcript content.

CORE REQUIREMENTS:
- ACCURACY: Every claim must be directly supported by the transcript
- LENGTH: Summary (150-400 words), Chapters (80-200 words each), Takeaways (3-8), Key Facts (3-6)
- TONE: Write in objective, article-like style (avoid "This video...", "The speaker...")

CONTENT STRUCTURE:
- TITLE: Descriptive, accurate (2-15 words)
- SUMMARY: Comprehensive overview of main points and structure
- TAKEAWAYS: Simple array of strings
- KEY FACTS: Simple array of strings
- KEYWORDS: Exactly 3 most relevant terms

CONTENT FILTERING:
- Remove all promotional content (speaker intros, calls-to-action, self-promotion)
- Keep only educational content about the strategy
- Correct obvious typos naturally

CHAPTER REQUIREMENTS:
"""

    # Add chapter-specific instructions if chapters are available
    if state.chapters:
        chapters_text = "\n".join([f"- {chapter['title']}" for chapter in state.chapters])
        base_prompt += f"""Use these video chapters as the basis for your breakdown:
{chapters_text}

- Create chapters that correspond to these sections
- Each chapter: header, summary (80-200 words), 3-6 key points
- Maintain logical flow"""
    else:
        base_prompt += """Create 4-8 thematic chapters based on content structure and topic transitions."""

    base_prompt += """

QUALITY CHECKS:
- Content matches transcript exactly (no external additions)
- All promotional content removed (intros, calls-to-action, self-promotion)
- Typos corrected naturally, meaning preserved
- Length balanced: substantial but not overwhelming
- Keywords highly relevant and searchable"""

    if state.enable_translation:
        language_name = state.target_language
        base_prompt += f"""

TRANSLATION:
- Translate to {language_name} with natural fluency
- Preserve technical terms and proper names
- Maintain same detail level and structure"""
    return base_prompt


def get_quality_prompt(state: GraphState) -> str:
    """Generate streamlined quality evaluation prompt."""
    base_prompt = """Evaluate the analysis on 8 aspects. Rate each "Fail", "Refine", or "Pass" with a specific reason.

ASPECTS:
1. TRANSCRIPT ACCURACY: Content directly supported by transcript (no external additions)
2. CONTENT LENGTH: Balanced length following guidelines (not too short/long)
3. COMPLETENESS: Entire content properly analyzed
4. STRUCTURE: Follows required schema perfectly with takeaways/key_facts as simple arrays
5. GRAMMAR: No typos/mistakes (semicolon usage in lists acceptable)
6. WRITING STYLE: Objective, article-like tone (no video references)
7. PROMOTIONAL REMOVAL: All promotional content completely removed
8. ARRAY VALIDITY: Takeaways/key_facts are valid arrays with appropriate content
9. KEYWORDS: Highly relevant and useful"""

    if state.enable_translation:
        language_name = state.target_language
        base_prompt += f"""
11. TRANSLATION QUALITY: Content is properly translated to {language_name} with natural fluency and maintained quality"""
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
    return base_prompt


def get_improvement_prompt(state: GraphState) -> str:
    """Generate streamlined improvement prompt."""
    base_prompt = """Improve the analysis based on quality feedback while maintaining transcript accuracy.

IMPROVEMENT PRIORITIES:
1. TRANSCRIPT ACCURACY: All content must be transcript-supported
2. PROMOTIONAL REMOVAL: Remove all intros, calls-to-action, self-promotion
3. LENGTH BALANCE: Follow guidelines (Summary: 150-400 words, Chapters: 80-200 each)
4. WRITING STYLE: Use objective, article-like tone (avoid "This video...", "The speaker...")
5. TYPO CORRECTION: Fix obvious typos naturally
6. ARRAY FORMATTING: Return takeaways/key_facts as simple string arrays

CONTENT TARGETS:
- Title: 2-15 words
- Summary: 150-400 words
- Chapters: 80-200 words each
- Takeaways: Simple array with 3-8 strings
- Key Facts: Simple array with 3-6 strings
- Keywords: Exactly 3"""

    if state.enable_translation:
        language_name = state.target_language
        base_prompt += f"""

TRANSLATION IMPROVEMENT REQUIREMENTS:
- Maintain all content in {language_name} ({state.target_language})
- Preserve translation quality while fixing identified issues
- Keep technical terms and proper names appropriate for the target language
- Maintain natural fluency and cultural relevance"""

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
        import json

        # Convert arrays to JSON strings for LLM processing
        takeaways_json = json.dumps(analysis.takeaways, ensure_ascii=False)
        key_facts_json = json.dumps(analysis.key_facts, ensure_ascii=False)

        return f"""# Improve this video analysis based on the following feedback:

## Quality Assessment (Total: {quality.total_score}/{quality.max_possible_score} - {quality.percentage_score}%):

### Completeness: {quality.completeness.rate}
{quality.completeness.reason}

### Structure: {quality.structure.rate}
{quality.structure.reason}

### Grammar: {quality.grammar.rate}
{quality.grammar.reason}

### No Garbage: {quality.no_garbage.rate}
{quality.no_garbage.reason}

### Correct Language: {quality.correct_language.rate}
{quality.correct_language.reason}

## Original Analysis:

### Title
{analysis.title}

### Summary
{analysis.summary}

### Key Takeaways (JSON format)
```json
{takeaways_json}
```

### Key Facts (JSON format)
```json
{key_facts_json}
```

## Improvement Instructions:
- Maintain the JSON format for takeaways and key facts in your response
- Each takeaway/fact should be a simple string
- Address the specific quality issues mentioned above
- Return takeaways and key_facts as valid JSON arrays

Please provide an improved version that addresses the specific issues identified above to improve the overall quality score. Format takeaways and key_facts as JSON arrays in your response."""

    @staticmethod
    def analysis_to_text(analysis: Analysis) -> str:
        """Convert analysis to text format for quality evaluation."""
        import json

        # Convert arrays to JSON strings for quality evaluation
        takeaways_json = json.dumps(analysis.takeaways, ensure_ascii=False)
        key_facts_json = json.dumps(analysis.key_facts, ensure_ascii=False)

        # Include detailed chapter information for completeness evaluation
        chapters_text = []
        for i, chapter in enumerate(analysis.chapters, 1):
            chapter_info = f"Chapter {i}: {chapter.header}"
            chapter_info += f"\n  Summary: {chapter.summary}"
            if chapter.key_points:
                chapter_info += f"\n  Key Points: {'; '.join(chapter.key_points)}"
            chapters_text.append(chapter_info)

        return f"""
Title: {analysis.title}
Summary: {analysis.summary}
Takeaways: {takeaways_json}
Key Facts: {key_facts_json}
Chapters: {chr(10).join(chapters_text)}
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

    # Prepare content with chapters information if available
    content = state.transcript_or_url
    if state.chapters:
        chapters_info = "\n\nVIDEO CHAPTERS:\n" + "\n".join([f"- {chapter['title']} (starts at {chapter['timeDescription']})" for chapter in state.chapters])
        content += chapters_info
        print(f"📋 Including {len(state.chapters)} video chapters in analysis")

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
    result: Analysis = chain.invoke({"content": content})

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
    print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
    print(f"Useful Keywords: {quality.useful_keywords.rate} - {quality.useful_keywords.reason}")
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

    # Prepare content parts
    content_parts = [
        types.Part(file_data=types.FileData(file_uri=state.transcript_or_url)),
    ]

    # Add chapters information if available
    if state.chapters:
        chapters_info = "VIDEO CHAPTERS:\n" + "\n".join([f"- {chapter['title']} (starts at {chapter['timeDescription']})" for chapter in state.chapters])
        content_parts.append(types.Part(text=chapters_info))
        print(f"📋 Including {len(state.chapters)} video chapters in Gemini analysis")

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
    print(f"No Garbage: {quality.no_garbage.rate} - {quality.no_garbage.reason}")
    print(f"Useful Keywords: {quality.useful_keywords.rate} - {quality.useful_keywords.reason}")
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
