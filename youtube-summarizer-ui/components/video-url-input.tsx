"use client"

import { Loader2 } from "lucide-react"; // Import Loader2 from lucide-react or wherever it's defined
import type React from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useMemo, useState } from "react";

interface VideoUrlInputProps {
  onSubmit: (url: string) => void
  isLoading: boolean
}

export function VideoUrlInput({ onSubmit, isLoading }: VideoUrlInputProps) {
  const [url, setUrl] = useState("")

  const isValidUrl = (value: string): boolean => {
    const v = value.trim()
    if (!v) return false
    try {
      // Basic absolute URL validation
      const parsed = new URL(v)
      // Optionally ensure it's a YouTube URL
      return /(youtube\.com|youtu\.be)/.test(parsed.hostname)
    } catch {
      return false
    }
  }

  const canSubmit = useMemo(() => isValidUrl(url), [url])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const v = url.trim()
    if (isValidUrl(v)) {
      onSubmit(v)
    }
  }

  return (
    <Card className="relative overflow-hidden bg-transparent border-0 shadow-none">
      <CardContent className="p-8 relative z-10">
        <form onSubmit={handleSubmit} className="flex gap-4">
          <Input
            type="text"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="flex-1"
            disabled={isLoading}
          />
          <Button
            type="submit"
            disabled={!canSubmit || isLoading}
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
