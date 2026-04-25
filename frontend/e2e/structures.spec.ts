/**
 * Phase 9 / Session 9.2 — Playwright structures spec.
 *
 * NOT in the default CI gate — needs uvicorn (port 8002) + Next.js
 * dev server + a seeded scientist@orion.dev user. Run locally:
 *
 *   uvicorn src.api.app:app --port 8002
 *   cd frontend && npm run dev
 *   cd frontend && npx playwright test e2e/structures.spec.ts
 *
 * The roadmap acceptance items (4):
 *   1. Upload Si.cif → preview shows a≈5.4307 Å, SG 227, 8 atoms.
 *   2. Confirm save → land on detail → 3D viewer renders 8 spheres.
 *   3. Export CIF → re-upload → MD5(parsed POSCAR) matches original.
 *   4. DataGrid: filter formula="Si" + density 2.0–3.0 → ≤ 5 rows.
 *
 * The MD5 round-trip (item 3) requires running the structure's
 * StructureMatcher equivalence on the backend — that's a backend
 * test, not a Playwright assertion. We instead verify the UI end of
 * the round-trip: download → upload-with-same-text → preview shows
 * the same formula + spacegroup + atom count.
 */

import { test, expect } from '@playwright/test'
import path from 'node:path'

const BASE = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:3000'
const SCIENTIST_EMAIL = process.env.ORION_TEST_SCIENTIST_EMAIL ?? 'scientist@orion.dev'
const SCIENTIST_PASS = process.env.ORION_TEST_SCIENTIST_PASS ?? 'change-me-in-seed-data'
const SI_FIXTURE = path.join(__dirname, '..', '..', 'tests', 'fixtures', 'mp_offline', 'Si.cif')

async function login(page) {
  await page.goto(`${BASE}/login`)
  await page.getByTestId('login-username').fill(SCIENTIST_EMAIL)
  await page.getByTestId('login-password').fill(SCIENTIST_PASS)
  await page.getByTestId('login-submit').click()
  await expect(page).toHaveURL(/\/dashboard/)
}

test.describe('Session 9.2 — structures UI', () => {
  test('upload Si.cif → preview shows correct lattice + spacegroup', async ({
    page,
  }) => {
    await login(page)
    await page.goto(`${BASE}/structures`)
    await page.getByTestId('structures-upload-button').click()
    await expect(page.getByTestId('upload-drawer')).toBeVisible()
    // The dropzone's hidden input accepts setInputFiles directly.
    await page.getByTestId('upload-input').setInputFiles(SI_FIXTURE)
    // Wait for the parse-preview drawer to populate.
    await expect(page.getByTestId('upload-preview')).toBeVisible({
      timeout: 5_000,
    })
    await expect(page.getByTestId('preview-formula')).toHaveText(/Si/)
    await expect(page.getByTestId('preview-atoms')).toHaveText('8')
    await expect(page.getByTestId('preview-spacegroup')).toContainText('227')
    // The diamond-Si conventional cell has a = 5.4307 Å (the value
    // pinned in the seed fixture).
    await expect(page.getByTestId('preview-abc')).toContainText(/5\.43/)
  })

  test('confirm save → detail page renders viewer', async ({ page }) => {
    await login(page)
    await page.goto(`${BASE}/structures`)
    await page.getByTestId('structures-upload-button').click()
    await page.getByTestId('upload-input').setInputFiles(SI_FIXTURE)
    await expect(page.getByTestId('upload-preview')).toBeVisible()
    await page.getByTestId('upload-confirm').click()
    // Detail page renders.
    await expect(page).toHaveURL(/\/structures\/[0-9a-f-]+/)
    await expect(page.getByTestId('detail-num-atoms')).toHaveText('8')
    await expect(page.getByTestId('detail-viewer')).toBeVisible()
    // The viewer's wrapper carries data-mesh-count = sphereCount + bondCount.
    // For the diamond Si conventional cell, 8 spheres + ~16 NN bonds ≥ 8.
    const meshCount = await page
      .getByTestId('detail-viewer')
      .getAttribute('data-mesh-count')
    expect(parseInt(meshCount ?? '0', 10)).toBeGreaterThanOrEqual(8)
  })

  test('export CIF → re-upload → preview matches original', async ({ page }) => {
    await login(page)
    await page.goto(`${BASE}/structures`)
    // Need an existing structure to export. Use the upload + save
    // path from the previous test as the setup.
    await page.getByTestId('structures-upload-button').click()
    await page.getByTestId('upload-input').setInputFiles(SI_FIXTURE)
    await expect(page.getByTestId('upload-preview')).toBeVisible()
    await page.getByTestId('upload-confirm').click()
    await expect(page).toHaveURL(/\/structures\/[0-9a-f-]+/)

    // Set the exporter to CIF (it's the default but be explicit).
    await page.getByTestId('export-format-select').click()
    await page.getByRole('option', { name: 'CIF' }).click()

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByTestId('export-button').click(),
    ])
    const downloadedPath = await download.path()
    expect(downloadedPath).toBeTruthy()

    // Re-upload the downloaded file via the same drawer flow.
    await page.goto(`${BASE}/structures`)
    await page.getByTestId('structures-upload-button').click()
    await page.getByTestId('upload-input').setInputFiles(downloadedPath!)
    await expect(page.getByTestId('upload-preview')).toBeVisible()
    await expect(page.getByTestId('preview-formula')).toHaveText(/Si/)
    await expect(page.getByTestId('preview-atoms')).toHaveText('8')
    await expect(page.getByTestId('preview-spacegroup')).toContainText('227')
  })

  test('DataGrid filters apply server-side', async ({ page }) => {
    await login(page)
    await page.goto(`${BASE}/structures`)
    // Type "Si" in the formula filter. The grid should re-fetch with
    // formula=Si in the query string.
    const requestPromise = page.waitForRequest(
      (req) => req.url().includes('/structures') && req.url().includes('formula=Si'),
      { timeout: 5_000 },
    )
    await page.getByTestId('filter-formula').fill('Si')
    await requestPromise
    // Then nudge the density slider — assert the next list request
    // includes density_min / density_max.
    const densityRequest = page.waitForRequest(
      (req) => req.url().includes('/structures') && /density_(min|max)=/.test(req.url()),
      { timeout: 5_000 },
    )
    // Grab the slider and shift focus then arrow it.
    await page.getByTestId('filter-density').focus()
    await page.keyboard.press('ArrowRight')
    await densityRequest
  })
})
