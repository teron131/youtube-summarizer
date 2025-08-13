"use client"

import type React from "react"
import { Loader2 } from "lucide-react" // Import Loader2 from lucide-react or wherever it's defined

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"

interface VideoUrlInputProps {
  onSubmit: (url: string) => void
  isLoading: boolean
}

export function VideoUrlInput({ onSubmit, isLoading }: VideoUrlInputProps) {
  const [url, setUrl] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (url.trim()) {
      onSubmit(url.trim())
    }
  }

  return (
    <Card className="relative overflow-hidden bg-transparent border-0 shadow-none">
      <CardContent className="p-8 relative z-10">
        <form onSubmit={handleSubmit} className="flex gap-4">
          <Input
            type="url"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="flex-1"
            disabled={isLoading}
          />
          <Button
            type="submit"
            disabled={!url.trim() || isLoading}
            className="bg-[#FF0000] hover:bg-[#CC0000] text-white"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Analyze"
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
