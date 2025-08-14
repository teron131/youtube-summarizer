"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { motion } from "framer-motion"
import { Loader2, Sparkles } from "lucide-react"

interface SummaryPanelProps {
  summary: string
  isLoading: boolean
  hasTranscription: boolean
}

export function SummaryPanel({ summary, isLoading, hasTranscription }: SummaryPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
    >
      <Card className="bg-transparent border-0 shadow-none">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Sparkles className="h-5 w-5 text-purple-500" />
              AI Summary
            </CardTitle>
            {isLoading && (
              <div className="text-sm text-muted-foreground flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Summarizing...
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {summary ? (
            <div className="h-64 w-full overflow-auto rounded-md border border-border/40 p-4">
              <div className="text-sm leading-relaxed">{summary}</div>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground border border-dashed border-border/40 rounded-md">
              <div className="text-center">
                <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>
                  {hasTranscription
                    ? 'Click "Generate Summary" to create an AI summary'
                    : "Generate transcription first to create a summary"}
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
