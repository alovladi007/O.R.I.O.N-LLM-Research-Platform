'use client'

import { Inter } from 'next/font/google'
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'

import { theme } from '@/lib/theme'
import { AppBar } from '@/components/layout/AppBar'
import { Footer } from '@/components/layout/Footer'

import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 1000 * 60 * 5, // 5 minutes
        retry: 3,
        refetchOnWindowFocus: false,
      },
    },
  }))

  return (
    <html lang="en">
      <head>
        <title>ORION Platform - AI-Driven Materials Science</title>
        <meta name="description" content="Revolutionizing materials science research with AI-powered discovery and simulation" />
      </head>
      <body className={inter.className}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <div className="flex flex-col min-h-screen">
              <AppBar />
              <main className="flex-grow">
                {children}
              </main>
              <Footer />
            </div>
          </ThemeProvider>
        </QueryClientProvider>
      </body>
    </html>
  )
}