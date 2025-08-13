"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { VideoUrlInput } from "./video-url-input"
import { VideoInfo } from "./video-info"
import { TranscriptionPanel } from "./transcription-panel"
import { SummaryPanel } from "./summary-panel"
import { ChatPanel } from "./chat-panel"
import type { VideoData } from "@/types/video"

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

      // Get video info
      const response = await fetch("/api/video-info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })

      if (!response.ok) throw new Error("Failed to fetch video info")

      const data = await response.json()
      setVideoData(data)
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
        body: JSON.stringify({ videoId: videoData.id }),
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
    if (!transcription) return

    setIsLoading(true)
    try {
      const response = await fetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcription }),
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

      <AnimatePresence mode="wait">
        {videoData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            {/* Video Information - Centered */}
            <div className="flex justify-center">
              <div className="w-full max-w-4xl">
                <VideoInfo data={videoData} />
              </div>
            </div>

            {/* Transcription and Summary - Split 50/50 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <TranscriptionPanel transcription={transcription} onTranscribe={handleTranscribe} isLoading={isLoading} />
              <SummaryPanel
                summary={summary}
                onSummarize={handleSummarize}
                isLoading={isLoading}
                hasTranscription={!!transcription}
              />
            </div>

            {/* Chat Panel - Full width */}
            {summary && (
              <div className="flex justify-center">
                <div className="w-full max-w-4xl">
                  <ChatPanel videoData={videoData} transcription={transcription} summary={summary} />
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
