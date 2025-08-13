import { type NextRequest, NextResponse } from "next/server"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

export async function POST(request: NextRequest) {
  try {
    const { message, context, history } = await request.json()

    const conversationHistory = history
      .map((msg: any) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`)
      .join("\n")

    const { text } = await generateText({
      model: openai("gpt-4o"),
      system: `You are a helpful AI assistant that answers questions about YouTube video content. You have access to the video's transcription and summary. Provide accurate, helpful responses based on the available information.

Video Context:
- Title: ${context.videoTitle}
- Summary: ${context.summary}
- Full Transcription: ${context.transcription}

Previous conversation:
${conversationHistory}`,
      prompt: message,
    })

    return NextResponse.json({ response: text })
  } catch (error) {
    console.error("Error in chat:", error)
    return NextResponse.json({ error: "Failed to generate response" }, { status: 500 })
  }
}
