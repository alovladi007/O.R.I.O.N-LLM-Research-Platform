/**
 * Phase 9 / Session 9.1 — Next.js edge route guard.
 *
 * Runs at the edge before every protected page renders. Reads the
 * ``orion_access_token`` cookie (httpOnly is fine — middleware
 * runs server-side, not in the browser). If absent, redirects to
 * ``/login?next=<original-path>`` so the user lands back where
 * they were after re-auth.
 *
 * The protected paths are listed in :data:`PROTECTED_PREFIXES`. To
 * gate a new page, add its top-level path here.
 *
 * This is a defense-in-depth layer on top of :func:`useRequireRole`
 * (which gates client-side after the bundle ships). A user with a
 * valid cookie skips this redirect; a user with no cookie or an
 * expired access cookie + valid refresh cookie still gets through
 * — the page mounts, the axios interceptor refreshes on first 401,
 * and the redux runs once more. That's by design: the middleware's
 * job is the lazy "is there *any* session token?" check, not the
 * exhaustive validation.
 */

import { NextRequest, NextResponse } from 'next/server'

const ACCESS_COOKIE = 'orion_access_token'
const REFRESH_COOKIE = 'orion_refresh_token'

const PROTECTED_PREFIXES = [
  '/dashboard',
  '/structures',
  '/jobs',
  '/workflows',
  '/campaigns',
  '/ml',
  '/agent',
]

function isProtected(pathname: string): boolean {
  return PROTECTED_PREFIXES.some(
    (p) => pathname === p || pathname.startsWith(`${p}/`),
  )
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl
  if (!isProtected(pathname)) return NextResponse.next()

  const access = request.cookies.get(ACCESS_COOKIE)?.value
  const refresh = request.cookies.get(REFRESH_COOKIE)?.value
  // Either cookie is enough to let the page mount — the client-side
  // refresh path will handle the rest. Only if BOTH are missing do
  // we redirect.
  if (access || refresh) return NextResponse.next()

  const next = encodeURIComponent(pathname + request.nextUrl.search)
  const loginUrl = new URL(`/login?next=${next}`, request.url)
  return NextResponse.redirect(loginUrl)
}

// Restrict the matcher so middleware doesn't run on static assets.
export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization)
     * - favicon.ico, robots.txt, sitemap.xml
     * - Anything with a file extension (.png, .jpg, .css, etc.)
     */
    '/((?!_next/static|_next/image|favicon\\.ico|robots\\.txt|sitemap\\.xml|.*\\.[^/]+$).*)',
  ],
}
