import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { url } = await request.json()

    const backendUrl = process.env.BACKEND_URL || "http://localhost:8080"
    const resp = await fetch(`${backendUrl}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, generate_summary: false }),
      cache: "no-store",
    })

    if (!resp.ok) {
      const text = await resp.text()
      throw new Error(`Backend error: ${resp.status} ${text}`)
    }

    const data = await resp.json()
    const transcription: string = data?.data?.subtitle || ""
    return NextResponse.json({ transcription })
  } catch (error) {
    console.error("Error transcribing via backend:", error)
    return NextResponse.json({ error: "Failed to transcribe video" }, { status: 500 })
  }
}
