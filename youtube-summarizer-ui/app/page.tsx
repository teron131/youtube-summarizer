import { VideoAnalyzer } from "@/components/video-analyzer"
import { ThemeToggle } from "@/components/theme-toggle"
import { Youtube } from "lucide-react"

export default function Page() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/40 bg-background/80 backdrop-blur-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-[#FF0000]">
              <Youtube className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-[#FF0000]">YouTube Analyzer</h1>
          </div>
          <ThemeToggle />
        </div>
      </header>
      <main className="container mx-auto px-4 py-8">
        <VideoAnalyzer />
      </main>
    </div>
  )
}
