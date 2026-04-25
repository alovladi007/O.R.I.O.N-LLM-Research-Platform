/**
 * Phase 9 / Session 9.1 — Playwright auth acceptance.
 *
 * NOT run in the default `tests.yml` CI gate — these tests need a
 * live backend (uvicorn on :8002) + the Next.js dev server. Run
 * locally with:
 *
 *   # terminal 1: backend
 *   uvicorn src.api.app:app --port 8002
 *   # terminal 2: frontend dev
 *   cd frontend && npm run dev
 *   # terminal 3: tests
 *   cd frontend && npx playwright test e2e/auth.spec.ts
 *
 * These specs pin the four roadmap acceptance items for 9.1:
 *   1. Log in as scientist@orion.dev → /dashboard, user menu shows email.
 *   2. Reload /dashboard → still authenticated (cookie survived).
 *   3. Token expiry → silent refresh on next request, no 401 bubble.
 *   4. Logout → cookies cleared, /dashboard → redirect to /login.
 *
 * The cookie-mode test that proves the JWT never reaches localStorage
 * is at the bottom — it's the security-critical assertion.
 *
 * Test fixtures depend on a seeded scientist@orion.dev user (Session
 * 1.5 seed_mp_subset.py creates this); the live backend must have
 * had that script run.
 */

import { test, expect } from '@playwright/test'

const BASE = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3000'

const SCIENTIST_EMAIL = process.env.ORION_TEST_SCIENTIST_EMAIL ?? 'scientist@orion.dev'
const SCIENTIST_PASS = process.env.ORION_TEST_SCIENTIST_PASS ?? 'change-me-in-seed-data'

test.describe('Session 9.1 — auth flow', () => {
  test('login lands on /dashboard with the scientist email visible', async ({ page }) => {
    await page.goto(`${BASE}/login`)
    await page.getByTestId('login-username').fill(SCIENTIST_EMAIL)
    await page.getByTestId('login-password').fill(SCIENTIST_PASS)
    await page.getByTestId('login-submit').click()
    await expect(page).toHaveURL(/\/dashboard/)
    await expect(page.getByTestId('dashboard-user-email')).toHaveText(SCIENTIST_EMAIL)
  })

  test('cookie survives reload — second visit to /dashboard does not redirect', async ({
    page,
    context,
  }) => {
    await page.goto(`${BASE}/login`)
    await page.getByTestId('login-username').fill(SCIENTIST_EMAIL)
    await page.getByTestId('login-password').fill(SCIENTIST_PASS)
    await page.getByTestId('login-submit').click()
    await expect(page).toHaveURL(/\/dashboard/)
    // Reload — the orion_access_token cookie must carry the session.
    await page.reload()
    await expect(page).toHaveURL(/\/dashboard/)
    // Cookie sanity: httpOnly is set; we can read its presence (not value)
    // via the context.
    const cookies = await context.cookies()
    const access = cookies.find((c) => c.name === 'orion_access_token')
    expect(access).toBeDefined()
    expect(access?.httpOnly).toBe(true)
    expect(access?.sameSite).toBe('Lax')
  })

  test('logout clears cookies and /dashboard redirects to /login', async ({
    page,
    context,
  }) => {
    await page.goto(`${BASE}/login`)
    await page.getByTestId('login-username').fill(SCIENTIST_EMAIL)
    await page.getByTestId('login-password').fill(SCIENTIST_PASS)
    await page.getByTestId('login-submit').click()
    await expect(page).toHaveURL(/\/dashboard/)
    await page.getByTestId('dashboard-logout').click()
    await expect(page).toHaveURL(/\/login/)
    const cookies = await context.cookies()
    expect(cookies.find((c) => c.name === 'orion_access_token')).toBeUndefined()
    expect(cookies.find((c) => c.name === 'orion_refresh_token')).toBeUndefined()
    // Direct nav to /dashboard now bounces back to /login.
    await page.goto(`${BASE}/dashboard`)
    await expect(page).toHaveURL(/\/login/)
  })

  test('access token never enters localStorage', async ({ page }) => {
    // The whole point of the cookie-mode rewrite. If a future
    // change re-introduces a localStorage write on login, this test
    // catches it.
    await page.goto(`${BASE}/login`)
    await page.getByTestId('login-username').fill(SCIENTIST_EMAIL)
    await page.getByTestId('login-password').fill(SCIENTIST_PASS)
    await page.getByTestId('login-submit').click()
    await expect(page).toHaveURL(/\/dashboard/)
    const ls = await page.evaluate(() => {
      const out: Record<string, string> = {}
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i)
        if (k) out[k] = localStorage.getItem(k) ?? ''
      }
      return out
    })
    // No key should contain a string starting with "ey" (JWT signature).
    for (const [key, val] of Object.entries(ls)) {
      expect(val.startsWith('ey'), `localStorage[${key}] looks like a JWT`).toBe(false)
    }
  })

  // The "5-second token expiry → silent refresh" acceptance from the
  // roadmap requires a backend test fixture that mints short-lived
  // tokens; that's a Session-9.1b backend addition (not exposed via
  // the live login endpoint today). The spec is parked here for
  // readability:
  test.skip('token expiry triggers silent refresh, no 401 bubble', async () => {
    /*
     * 1. Log in via a backend fixture that mints a JWT with
     *    access_token_expire_minutes = 5/60 (5 seconds).
     * 2. Wait 6 s.
     * 3. Navigate to /structures → axios fires GET /structures →
     *    server returns 401 → refresh interceptor calls
     *    /auth/refresh?mode=cookie → retries the original request →
     *    page renders the structures grid.
     * 4. Assertion: the network log shows exactly one /auth/refresh
     *    call, the page does NOT bounce to /login.
     */
  })
})
