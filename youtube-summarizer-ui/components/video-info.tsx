"use client"

import { motion } from "framer-motion"
import { Play, Clock, Eye, ThumbsUp, Calendar } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { VideoData } from "@/types/video"

interface VideoInfoProps {
  data: VideoData
}

export function VideoInfo({ data }: VideoInfoProps) {
  const formatDuration = (duration: string) => {
    // Convert ISO 8601 duration to readable format
    const match = duration.match(/PT(\d+H)?(\d+M)?(\d+S)?/)
    if (!match) return duration

    const hours = match[1] ? Number.parseInt(match[1]) : 0
    const minutes = match[2] ? Number.parseInt(match[2]) : 0
    const seconds = match[3] ? Number.parseInt(match[3]) : 0

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`
    }
    return `${minutes}:${seconds.toString().padStart(2, "0")}`
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.5 }}>
      <Card className="overflow-hidden bg-transparent border-0 shadow-none">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Play className="h-5 w-5 text-red-500" />
            Video Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-6">
            {/* Thumbnail - Left Side */}
            <div className="flex-shrink-0">
              <div className="w-80 aspect-video rounded-lg overflow-hidden bg-muted">
                <img
                  src={data.thumbnail || "/placeholder.svg"}
                  alt={data.title}
                  className="w-full h-full object-cover"
                />
              </div>
            </div>

            {/* Video Info - Right Side */}
            <div className="flex-1 space-y-4">
              <div>
                <h3 className="font-semibold text-xl mb-2 line-clamp-2">{data.title}</h3>
                <p className="text-sm text-muted-foreground mb-3">{data.channelTitle}</p>

                <div className="flex flex-wrap gap-2 mb-4">
                  <Badge variant="secondary" className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {formatDuration(data.duration)}
                  </Badge>
                  <Badge variant="secondary" className="flex items-center gap-1">
                    <Eye className="h-3 w-3" />
                    {formatNumber(data.viewCount)} views
                  </Badge>
                  <Badge variant="secondary" className="flex items-center gap-1">
                    <ThumbsUp className="h-3 w-3" />
                    {formatNumber(data.likeCount)}
                  </Badge>
                  <Badge variant="secondary" className="flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {new Date(data.publishedAt).toLocaleDateString()}
                  </Badge>
                </div>

                {data.description && (
                  <p className="text-sm text-muted-foreground line-clamp-4 leading-relaxed">{data.description}</p>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
