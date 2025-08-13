import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { url } = await request.json()

    // Extract video ID from YouTube URL
    const videoId = extractVideoId(url)
    if (!videoId) {
      return NextResponse.json({ error: "Invalid YouTube URL" }, { status: 400 })
    }

    // Mock response - replace with actual YouTube API call
    const videoData = {
      id: videoId,
      title: "Sample Video Title - Replace with YouTube API",
      description: "This is a sample description. In production, this would come from the YouTube Data API v3.",
      channelTitle: "Sample Channel",
      thumbnail: `/placeholder.svg?height=180&width=320&query=youtube+video+thumbnail`,
      duration: "PT10M30S",
      viewCount: 1234567,
      likeCount: 12345,
      publishedAt: new Date().toISOString(),
      url: url,
    }

    return NextResponse.json(videoData)
  } catch (error) {
    console.error("Error fetching video info:", error)
    return NextResponse.json({ error: "Failed to fetch video info" }, { status: 500 })
  }
}

function extractVideoId(url: string): string | null {
  const regex = /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/
  const match = url.match(regex)
  return match ? match[1] : null
}
