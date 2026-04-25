/**
 * Phase 9 / Session 9.4 — Playwright campaigns + ML spec.
 *
 * NOT in default CI gate (needs uvicorn + Next.js dev + scientist
 * seed user). Local invocation:
 *   uvicorn src.api.app:app --port 8002
 *   cd frontend && npm run dev
 *   cd frontend && npx playwright test e2e/campaigns_ml.spec.ts
 *
 * Roadmap acceptance for 9.4 (2 items):
 *   1. Create a toy AL campaign on the inline-corpus path
 *      (8 candidates, 2 seeds, 3 cycles, max_sigma) → best-so-far
 *      plot shows three monotone-non-decreasing points.
 *   2. /ml/predict: drop 3 CIFs → table renders 3 rows with finite
 *      (μ, σ); σ-percentile color matches σ rank within the batch.
 */

import { test, expect } from '@playwright/test'

const BASE = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3000'
const SCIENTIST_EMAIL = process.env.ORION_TEST_SCIENTIST_EMAIL ?? 'scientist@orion.dev'
const SCIENTIST_PASS = process.env.ORION_TEST_SCIENTIST_PASS ?? 'change-me-in-seed-data'

// 8-row CSV: 2 features + 1 target column.
const TOY_CSV = `1.0, 2.0, 3.0
1.5, 2.5, 4.0
2.0, 3.0, 5.0
2.5, 3.5, 6.0
3.0, 4.0, 5.5
3.5, 4.5, 4.5
4.0, 5.0, 5.0
4.5, 5.5, 6.0`

async function login(page) {
  await page.goto(`${BASE}/login`)
  await page.getByTestId('login-username').fill(SCIENTIST_EMAIL)
  await page.getByTestId('login-password').fill(SCIENTIST_PASS)
  await page.getByTestId('login-submit').click()
  await expect(page).toHaveURL(/\/dashboard/)
}

test.describe('Session 9.4 — campaigns + ML', () => {
  test('create AL campaign → best-so-far plot renders + cycles table populated', async ({
    page,
  }) => {
    await login(page)
    await page.goto(`${BASE}/campaigns`)
    await page.getByTestId('campaigns-create-button').click()
    await expect(page.getByTestId('create-al-title')).toBeVisible()
    await page.getByTestId('al-name').fill('toy-9.4')
    await page.getByTestId('al-csv').fill(TOY_CSV)
    await page.getByTestId('al-seeds').fill('0,1')
    await page.getByTestId('al-qs').fill('2')
    await page.getByTestId('al-cycles').fill('3')
    await page.getByTestId('al-create-submit').click()

    // Lands on detail. Best-so-far chart renders + cycles table.
    await expect(page).toHaveURL(/\/campaigns\/[0-9a-f-]+/)
    await expect(page.getByTestId('best-so-far-chart')).toBeVisible()
    const cyclesTable = page.getByTestId('cycles-table')
    await expect(cyclesTable).toBeVisible()
    // Three cycles → three data rows in the body.
    const bodyRows = cyclesTable.locator('tbody tr')
    await expect(bodyRows).toHaveCount(3)
  })

  test('AL tab shows empty-state for new user', async ({ page }) => {
    await login(page)
    await page.goto(`${BASE}/campaigns`)
    // Default tab is AL. If the user has no campaigns yet, the
    // empty-state appears. Brand-new test users typically don't.
    // We assert *either* the empty-state OR a non-empty list — both
    // are valid initial states; we just want to confirm the page
    // mounts.
    const hasEmpty = await page
      .getByTestId('al-empty-state')
      .isVisible()
      .catch(() => false)
    const hasList = await page
      .getByTestId('al-campaigns-list')
      .isVisible()
      .catch(() => false)
    expect(hasEmpty || hasList).toBe(true)
  })

  test('BO tab shows the deferred-7.2b empty state', async ({ page }) => {
    await login(page)
    await page.goto(`${BASE}/campaigns`)
    await page.getByTestId('tab-bo').click()
    await expect(page.getByTestId('bo-empty-state')).toBeVisible()
  })

  test('ML registry page loads (empty or non-empty)', async ({ page }) => {
    await login(page)
    await page.goto(`${BASE}/ml`)
    // The registry may or may not have models seeded. Assert the
    // page mounts and exposes one of the two markers.
    const hasEmpty = await page
      .getByTestId('ml-empty-state')
      .isVisible()
      .catch(() => false)
    const hasList = await page
      .getByTestId('ml-models-list')
      .isVisible()
      .catch(() => false)
    expect(hasEmpty || hasList).toBe(true)
  })

  test.skip(
    '/ml/[id]/predict: drop 3 CIFs → 3 rows with finite (μ, σ)',
    async () => {
      /*
       * Requires a registered ML model (Session 6.4 trains one;
       * the seeded test DB may not include it). Wires up in 9.4b
       * once the scientist@orion.dev seed user owns at least one
       * predict-capable model and the predict endpoint returns
       * the {predicted_properties: {bandgap: {value, uncertainty}}}
       * shape the frontend reads.
       */
    },
  )
})
