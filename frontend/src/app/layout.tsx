import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { AppRouterCacheProvider } from '@mui/material-nextjs/v14-appRouter'
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { Toaster } from 'react-hot-toast'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

import { theme } from '@/lib/theme'
import { AuthProvider } from '@/contexts/AuthContext'
import { WebSocketProvider } from '@/contexts/WebSocketContext'
import { NotificationProvider } from '@/contexts/NotificationContext'
import { AppBar } from '@/components/layout/AppBar'
import { Footer } from '@/components/layout/Footer'
import { ProgressBar } from '@/components/common/ProgressBar'

import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ORION Platform - AI-Driven Materials Science',
  description: 'Revolutionizing materials science research with AI-powered discovery and simulation',
  keywords: 'materials science, AI, simulation, nanomaterials, research platform',
  authors: [{ name: 'ORION Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#1976d2',
  manifest: '/manifest.json',
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
  openGraph: {
    title: 'ORION Platform',
    description: 'AI-Driven Materials Science Platform',
    url: 'https://orion-platform.ai',
    siteName: 'ORION',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'ORION Platform',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ORION Platform',
    description: 'AI-Driven Materials Science Platform',
    images: ['/twitter-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 10, // 10 minutes
      retry: 3,
      refetchOnWindowFocus: false,
    },
  },
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <AppRouterCacheProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <QueryClientProvider client={queryClient}>
              <AuthProvider>
                <WebSocketProvider>
                  <NotificationProvider>
                    <ProgressBar />
                    <div className="flex flex-col min-h-screen">
                      <AppBar />
                      <main className="flex-grow">
                        {children}
                      </main>
                      <Footer />
                    </div>
                    <Toaster
                      position="top-right"
                      toastOptions={{
                        duration: 4000,
                        style: {
                          background: '#363636',
                          color: '#fff',
                        },
                        success: {
                          duration: 3000,
                          iconTheme: {
                            primary: '#4caf50',
                            secondary: '#fff',
                          },
                        },
                        error: {
                          duration: 5000,
                          iconTheme: {
                            primary: '#f44336',
                            secondary: '#fff',
                          },
                        },
                      }}
                    />
                  </NotificationProvider>
                </WebSocketProvider>
              </AuthProvider>
              <ReactQueryDevtools initialIsOpen={false} />
            </QueryClientProvider>
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  )
}