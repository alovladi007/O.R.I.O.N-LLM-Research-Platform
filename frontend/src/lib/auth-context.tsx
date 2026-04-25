'use client'

/**
 * Phase 9 / Session 9.1 — AuthContext + useAuth() / useRequireRole().
 *
 * Holds the *in-memory* user state on the client. The auth tokens
 * themselves live in httpOnly cookies (set by the backend at
 * /auth/login?mode=cookie); JS never touches them. The context is
 * populated by:
 *
 *   1. ``RootLayout`` mounts the provider, which immediately fires
 *      ``GET /auth/me``. If the cookies are valid, we get a user
 *      back and the context settles to ``{user, loading: false}``.
 *      If 401, the axios interceptor's refresh path may or may not
 *      recover; either way, the context lands at ``{user: null}``.
 *
 *   2. ``login()`` POSTs cookie-mode credentials, then refetches
 *      ``/auth/me``. Pages call ``router.push(next)`` after.
 *
 *   3. ``logout()`` POSTs ``/auth/logout`` (clears cookies
 *      server-side) and resets the context.
 *
 * The reason the context owns the ``user`` cache rather than letting
 * react-query do it: a page that needs to know "is the user logged
 * in?" inside the render of a header / nav-bar shouldn't trigger a
 * suspense boundary or block on a query. AuthContext gives that
 * synchronous answer.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { useRouter } from 'next/navigation'

import { api, type LoginRequest, type UserResponse } from './api'

interface AuthContextValue {
  user: UserResponse | null
  loading: boolean
  login: (body: LoginRequest) => Promise<UserResponse>
  logout: () => Promise<void>
  refresh: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const u = await api.auth.me()
      setUser(u)
    } catch {
      setUser(null)
    } finally {
      setLoading(false)
    }
  }, [])

  // Eagerly probe /auth/me on mount. If the user holds a valid
  // session cookie from a previous visit, this populates the context
  // before any page renders — the route guard in ``middleware.ts``
  // also checks the cookie at the edge for non-authed paths.
  useEffect(() => {
    void refresh()
  }, [refresh])

  const login = useCallback(
    async (body: LoginRequest) => {
      const token = await api.auth.login(body)
      // The login response carries the user; trust it instead of a
      // round-trip /me call.
      setUser(token.user)
      setLoading(false)
      return token.user
    },
    [],
  )

  const logout = useCallback(async () => {
    await api.auth.logout()
    setUser(null)
    router.push('/login')
  }, [router])

  const value = useMemo<AuthContextValue>(
    () => ({ user, loading, login, logout, refresh }),
    [user, loading, login, logout, refresh],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (ctx === undefined) {
    throw new Error('useAuth must be used inside an <AuthProvider>')
  }
  return ctx
}

/**
 * Role-gating hook for protected pages.
 *
 * Usage:
 *
 *   export default function AdminPage() {
 *     useRequireRole(['admin'])  // redirects to /login if not allowed
 *     return <PageContent />
 *   }
 *
 * Behavior:
 *   - While ``loading`` is true, no redirect happens (we don't know
 *     yet whether the user satisfies the role).
 *   - If the user is null (logged out), redirect to
 *     ``/login?next=<pathname>``.
 *   - If the user exists but their role isn't in ``allowed``, redirect
 *     to ``/dashboard`` (the "you're logged in but not allowed here"
 *     fallback). Caller can pass a different ``fallback`` if needed.
 */
export function useRequireRole(
  allowed: string[],
  fallback: string = '/dashboard',
): { user: UserResponse | null; loading: boolean } {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (loading) return
    if (!user) {
      const next =
        typeof window !== 'undefined'
          ? encodeURIComponent(window.location.pathname + window.location.search)
          : ''
      router.push(`/login?next=${next}`)
      return
    }
    if (!allowed.includes(user.role)) {
      router.push(fallback)
    }
  }, [user, loading, allowed, fallback, router])

  return { user, loading }
}
