import { type NextRequest, NextResponse } from "next/server"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

export async function POST(request: NextRequest) {
  try {
    const { transcription } = await request.json()

    const { text } = await generateText({
      model: openai("gpt-4o"),
      system:
        "You are an expert at summarizing video content. Create concise, informative summaries that capture the key points and main themes.",
      prompt: `Please summarize the following video transcription in 3-4 paragraphs, highlighting the main points, key insights, and important takeaways:

${transcription}`,
    })

    return NextResponse.json({ summary: text })
  } catch (error) {
    console.error("Error summarizing:", error)
    return NextResponse.json({ error: "Failed to generate summary" }, { status: 500 })
  }
}
