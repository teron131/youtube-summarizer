"""
Test script using YouTube Summarizer with LangGraph verification workflow.

Environment Variables Required:
    - SCRAPECREATORS_API_KEY: Your Scrape Creators API key
    - OPENROUTER_API_KEY: Your OpenRouter API key (or GEMINI_API_KEY)

Usage:
    uv run python test.py [youtube_url]

Examples:
    uv run python test.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
"""

import sys
import time

from dotenv import load_dotenv

from youtube_summarizer.summarizer import summarize_video
from youtube_summarizer.utils import is_youtube_url

load_dotenv()


# ============================================================================
# Main
# ============================================================================


def main():
    """Main test function."""
    test_url = "https://youtu.be/pmdiKAE_GLs"

    if len(sys.argv) > 1:
        test_url = sys.argv[1]

    print("=" * 80)
    print("TEST: YouTube Summarizer with LangGraph Verification")
    print("=" * 80)
    print(f"📹 Video URL: {test_url}\n")

    try:
        # Step 1: Get transcript (or use URL directly)
        if is_youtube_url(test_url):
            # Use URL directly - summarize_video will handle transcript extraction
            content = test_url
        else:
            # Assume it's a transcript
            content = test_url

        # Step 2: Analyze with LangGraph workflow
        print("\n" + "=" * 80)
        print("ANALYSIS & VERIFICATION WORKFLOW")
        print("=" * 80)

        start_time = time.time()
        analysis = summarize_video(content)
        elapsed = time.time() - start_time

        # Step 3: Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\n⏱️  Total time: {elapsed:.2f}s")
        print(f"\n📝 Title: {analysis.title}")
        print(f"\n📝 Summary:")
        print(f"{analysis.summary}")
        print(f"\n🎯 Takeaways ({len(analysis.takeaways)}):")
        for i, takeaway in enumerate(analysis.takeaways, 1):
            print(f"  {i}. {takeaway}")
        print(f"\n📚 Chapters ({len(analysis.chapters)}):")
        for i, chapter in enumerate(analysis.chapters, 1):
            print(f"  {i}. {chapter.header}")
            print(f"     {chapter.summary[:100]}...")
        print(f"\n🔑 Keywords: {', '.join(analysis.keywords)}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
