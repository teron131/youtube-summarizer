"use client"

import { motion } from "framer-motion"
import { Sparkles, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"

interface SummaryPanelProps {
  summary: string
  onSummarize: () => void
  isLoading: boolean
  hasTranscription: boolean
}

export function SummaryPanel({ summary, onSummarize, isLoading, hasTranscription }: SummaryPanelProps) {
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
            <Button
              onClick={onSummarize}
              disabled={!hasTranscription || isLoading}
              size="sm"
              className="bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Summarizing...
                </>
              ) : (
                "Generate Summary"
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {summary ? (
            <ScrollArea className="h-64 w-full rounded-md border border-border/40 p-4">
              <div className="text-sm leading-relaxed">{summary}</div>
            </ScrollArea>
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
