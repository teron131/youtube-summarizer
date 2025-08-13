"use client"

import { motion } from "framer-motion"
import { FileText, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"

interface TranscriptionPanelProps {
  transcription: string
  onTranscribe: () => void
  isLoading: boolean
}

export function TranscriptionPanel({ transcription, onTranscribe, isLoading }: TranscriptionPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card className="bg-transparent border-0 shadow-none">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg">
              <FileText className="h-5 w-5 text-blue-500" />
              Transcription
            </CardTitle>
            <Button
              onClick={onTranscribe}
              disabled={isLoading}
              size="sm"
              className="bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Transcribing...
                </>
              ) : (
                "Generate Transcription"
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {transcription ? (
            <ScrollArea className="h-64 w-full rounded-md border border-border/40 p-4">
              <div className="text-sm leading-relaxed whitespace-pre-wrap">{transcription}</div>
            </ScrollArea>
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground border border-dashed border-border/40 rounded-md">
              <div className="text-center">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Click "Generate Transcription" to extract audio text</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
