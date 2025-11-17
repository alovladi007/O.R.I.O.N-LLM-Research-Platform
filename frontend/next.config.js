/** @type {import('next').NextConfig} */

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
})

const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Internationalization
  i18n: {
    locales: ['en', 'es', 'fr', 'de', 'zh', 'ja'],
    defaultLocale: 'en',
  },
  
  // Image optimization
  images: {
    domains: [
      'localhost',
      'api.orion-platform.ai',
      'storage.googleapis.com',
      's3.amazonaws.com',
      'avatars.githubusercontent.com',
    ],
    formats: ['image/avif', 'image/webp'],
  },
  
  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8002',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8002',
    NEXT_PUBLIC_APP_NAME: 'ORION Platform',
    NEXT_PUBLIC_APP_VERSION: '2.0.0',
  },
  
  // Headers for security
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()'
          },
        ],
      },
    ]
  },
  
  // Redirects
  async redirects() {
    return [
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
      {
        source: '/api',
        destination: '/api/v1',
        permanent: true,
      },
    ]
  },
  
  // Rewrites for API proxy
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8002'}/api/v1/:path*`,
      },
    ]
  },
  
  // Webpack configuration
  webpack: (config, { isServer }) => {
    // Fix for canvas in Node.js environment
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      }
    }
    
    // Add SVG support
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    })
    
    return config
  },
  
  // Experimental features
  experimental: {},

  // Performance monitoring
  analyticsId: process.env.NEXT_PUBLIC_ANALYTICS_ID,
  
  // Compiler options
  compiler: {
    // Remove console logs in production
    removeConsole: process.env.NODE_ENV === 'production' ? {
      exclude: ['error', 'warn'],
    } : false,
    // Enable emotion
    emotion: true,
  },
  
  // Output configuration
  output: 'standalone',
  
  // TypeScript
  typescript: {
    ignoreBuildErrors: false,
  },
  
  // ESLint
  eslint: {
    ignoreDuringBuilds: false,
  },
}

module.exports = withBundleAnalyzer(nextConfig)