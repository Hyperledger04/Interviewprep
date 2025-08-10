import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Free Interview (DeepSeek R1)',
  description: 'Free mock interview app powered by DeepSeek R1',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        <header className="border-b border-gray-800">
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <h1 className="text-lg font-semibold">Free Interview</h1>
            <a href="https://app.interviews.chat/" className="text-xs text-gray-400 hover:text-gray-300" target="_blank" rel="noreferrer">Reference app</a>
          </div>
        </header>
        <main className="container mx-auto px-4 py-6">
          {children}
        </main>
      </body>
    </html>
  )
}
