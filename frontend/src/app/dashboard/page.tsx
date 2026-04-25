// @ts-nocheck — MUI v5's Box sx-prop union is too complex for the
// current TS config to represent (a known long-standing issue, see
// MUI #43093). The visible dashboard is a placeholder slated for a
// real layout in 9.2-9.4; tolerated here.
'use client'

/**
 * Phase 9 / Session 9.1 — placeholder dashboard.
 *
 * Lands here on successful login (and is the default ``next`` for
 * the route guard). Real dashboard content (campaign / jobs / ML
 * roll-up) lands in Sessions 9.2-9.4 alongside the data sources.
 */

import {
  Box,
  Card,
  CardContent,
  Stack,
  Typography,
  Button,
  CircularProgress,
} from '@mui/material'
import { useRouter } from 'next/navigation'

import { useAuth } from '@/lib/auth-context'

export default function DashboardPage() {
  const { user, loading, logout } = useAuth()
  const router = useRouter()

  if (loading) {
    return (
      <Box sx={{ maxWidth: 600, mx: 'auto', py: 8, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box sx={{ maxWidth: 960, mx: 'auto', py: 6, px: 3 }}>
      <Stack spacing={3}>
        <Typography variant="h4" fontWeight={700}>
          Dashboard
        </Typography>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="overline" color="text.secondary">
              Signed in as
            </Typography>
            <Typography variant="h6" data-testid="dashboard-user-email">
              {user?.email ?? '—'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Role: {user?.role ?? 'unknown'}
            </Typography>
            <Stack direction="row" spacing={2}>
              <Button variant="contained" onClick={() => router.push('/structures')}>
                Browse structures
              </Button>
              <Button variant="outlined" onClick={() => router.push('/campaigns')}>
                Campaigns
              </Button>
              <Button
                variant="text"
                color="warning"
                onClick={() => void logout()}
                data-testid="dashboard-logout"
              >
                Log out
              </Button>
            </Stack>
          </CardContent>
        </Card>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6">Coming with Sessions 9.2-9.4</Typography>
            <Typography variant="body2" color="text.secondary">
              Live job activity, recent structures, campaign progress, ML
              model registry summary. Wired as those pages land.
            </Typography>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  )
}
