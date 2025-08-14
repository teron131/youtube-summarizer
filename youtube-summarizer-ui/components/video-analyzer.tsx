"use client"

import type { VideoData } from "@/types/video"
import { AnimatePresence, motion } from "framer-motion"
import { useState } from "react"
import { ChatPanel } from "./chat-panel"
import { SummaryPanel } from "./summary-panel"
import { TranscriptionPanel } from "./transcription-panel"
import { VideoInfo } from "./video-info"
import { VideoUrlInput } from "./video-url-input"

export function VideoAnalyzer() {
  const [videoData, setVideoData] = useState<VideoData | null>(null)
  const [transcription, setTranscription] = useState<string>("")
  const [summary, setSummary] = useState<string>("")
  const [isLoading, setIsLoading] = useState(false)

  const handleVideoSubmit = async (url: string) => {
    setIsLoading(true)
    try {
      // Reset previous data
      setVideoData(null)
      setTranscription("")
      setSummary("")

      // Step 1: Video info
      const infoResp = await fetch("/api/video-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      if (!infoResp.ok) throw new Error("Failed to fetch video info")
      const info = await infoResp.json()
      setVideoData(info)

      // Step 2: Transcription
      const transResp = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      if (!transResp.ok) throw new Error("Failed to transcribe video")
      const transData = await transResp.json()
      const transcript = (transData.transcription as string) || ""
      setTranscription(transcript)

      // Step 3: Summary
      const sumResp = await fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcription: transcript, url, title: info.title, author: info.channelTitle }),
      })
      if (!sumResp.ok) throw new Error("Failed to summarize")
      const sumData = await sumResp.json()
      setSummary(sumData.summary || "")
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleTranscribe = async () => {
    if (!videoData) return

    setIsLoading(true)
    try {
      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoData.url }),
      })

      if (!response.ok) throw new Error("Failed to transcribe video")

      const data = await response.json()
      setTranscription(data.transcription)
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSummarize = async () => {
    if (!videoData) return

    setIsLoading(true)
    try {
      const response = await fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcription, url: videoData.url, title: videoData.title, author: videoData.channelTitle }),
      })

      if (!response.ok) throw new Error("Failed to summarize")

      const data = await response.json()
      setSummary(data.summary)
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <VideoUrlInput onSubmit={handleVideoSubmit} isLoading={isLoading} />

      <div className="space-y-8">
        {/* Always render panels; progressively fill content */}
        <div className="flex justify-center">
          <div className="w-full max-w-4xl">
            <AnimatePresence mode="wait">
              {videoData ? (
                <motion.div
                  key="video-info"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <VideoInfo data={videoData} />
                </motion.div>
              ) : (
                <motion.div
                  key="video-placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="rounded-lg border border-dashed border-border/40 p-8 text-center text-muted-foreground"
                >
                  Paste a YouTube URL above and click Analyze to load video details.
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <TranscriptionPanel transcription={transcription} isLoading={isLoading} />
          <SummaryPanel summary={summary} isLoading={isLoading} hasTranscription={!!transcription} />
        </div>

        {summary && (
          <div className="flex justify-center">
            <div className="w-full max-w-4xl">
              <ChatPanel videoData={videoData!} transcription={transcription} summary={summary} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
