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
    // Map backend response to UI VideoData
    const videoData = {
      id: data?.data?.url ?? url,
      title: data?.data?.title ?? "Unknown",
      description: "",
      channelTitle: data?.data?.author ?? "Unknown",
      thumbnail: "/placeholder.svg?height=180&width=320",
      duration: data?.data?.processing_time ?? "",
      viewCount: 0,
      likeCount: 0,
      publishedAt: new Date().toISOString(),
      url,
    }

    return NextResponse.json(videoData)
  } catch (error) {
    console.error("Error fetching video info via backend:", error)
    return NextResponse.json({ error: "Failed to fetch video info" }, { status: 500 })
  }
}
