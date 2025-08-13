"use client"

export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  return (
    <html>
      <body>
        <div style={{padding: 24}}>
          <h1>App Error</h1>
          <pre style={{whiteSpace: 'pre-wrap'}}>{String(error)}</pre>
        </div>
      </body>
    </html>
  )
}


