import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { transcription, url, title, author } = await request.json()

    const backendUrl = process.env.BACKEND_URL || "http://localhost:8080"
    const payload = {
      url,
      generate_summary: true,
    }

    // If transcription already exists, send it along to reduce backend work
    if (transcription) {
      Object.assign(payload as any, { transcription, title, author })
    }

    const resp = await fetch(`${backendUrl}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    })

    if (!resp.ok) {
      const text = await resp.text()
      throw new Error(`Backend error: ${resp.status} ${text}`)
    }

    const data = await resp.json()
    const summary: string | null = data?.data?.summary ?? null
    return NextResponse.json({ summary })
  } catch (error) {
    console.error("Error summarizing via backend:", error)
    return NextResponse.json({ error: "Failed to generate summary" }, { status: 500 })
  }
}
