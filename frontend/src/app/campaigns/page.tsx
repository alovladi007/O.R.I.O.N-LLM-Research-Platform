// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.4 — /campaigns list page.
 *
 * Lists active-learning campaigns from /api/v1/al/campaigns
 * (Session 6.5). BO campaigns will land alongside Session 7.2b's
 * DB promotion of campaigns_v2 — until then the BO tab shows an
 * empty state with a deep link to /bo/suggest. The roadmap
 * explicitly calls out this dual rendering ("until then, render
 * only the AL list and stub a 'BO campaigns coming with 7.2b'
 * empty state").
 */

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  IconButton,
  Stack,
  Tab,
  Tabs,
  Typography,
} from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { Add, Refresh } from '@mui/icons-material'

import {
  api,
  formatErrorMessage,
  useRequireRole,
  type ALCampaignResponse,
} from '@/lib/api'
import { CreateALCampaignDialog } from '@/components/campaigns/CreateALCampaignDialog'

export default function CampaignsListPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const router = useRouter()
  const [tab, setTab] = useState(0)
  const [createOpen, setCreateOpen] = useState(false)

  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['al-campaigns'],
    queryFn: () => api.al.list(),
    refetchInterval: 5_000,
  })

  return (
    <Box sx={{ maxWidth: 1300, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, flexGrow: 1 }}>
          Campaigns
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => refetch()}
        >
          Refresh
        </Button>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateOpen(true)}
          data-testid="campaigns-create-button"
        >
          New campaign
        </Button>
      </Stack>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="Active learning" data-testid="tab-al" />
        <Tab label="Bayesian optimization" data-testid="tab-bo" />
      </Tabs>

      {tab === 0 && (
        <ALCampaignsList
          isLoading={isLoading}
          error={error}
          campaigns={data ?? []}
          onOpen={(id) => router.push(`/campaigns/${id}`)}
        />
      )}

      {tab === 1 && (
        <Alert severity="info" data-testid="bo-empty-state">
          BO campaigns surface here once Session 7.2b promotes the
          in-memory campaigns_v2 store to the DB. The BO engine itself
          is live today — call <code>POST /api/v1/bo/suggest</code>
          directly (or via <code>api.bo.suggest()</code>) for ad-hoc
          one-shot use.
        </Alert>
      )}

      <CreateALCampaignDialog
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={(id) => {
          setCreateOpen(false)
          router.push(`/campaigns/${id}`)
        }}
      />
    </Box>
  )
}

function ALCampaignsList({
  isLoading,
  error,
  campaigns,
  onOpen,
}: {
  isLoading: boolean
  error: unknown
  campaigns: ALCampaignResponse[]
  onOpen: (id: string) => void
}) {
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    )
  }
  if (error) {
    return <Alert severity="error">{formatErrorMessage(error)}</Alert>
  }
  if (!campaigns.length) {
    return (
      <Alert severity="info" data-testid="al-empty-state">
        No AL campaigns yet. Click “New campaign” to start one.
      </Alert>
    )
  }
  return (
    <Stack spacing={2} data-testid="al-campaigns-list">
      {campaigns.map((c) => {
        const final = c.final_val_mae
        return (
          <Card
            key={c.id as string}
            variant="outlined"
            sx={{ cursor: 'pointer' }}
            onClick={() => onOpen(c.id as string)}
            data-testid={`campaign-card-${c.id}`}
          >
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={2}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  {c.name}
                </Typography>
                <Chip
                  label={`${c.n_cycles_completed} cycle${c.n_cycles_completed === 1 ? '' : 's'}`}
                  size="small"
                />
                {final != null && (
                  <Chip
                    label={`val MAE ${final.toFixed(3)}`}
                    color="success"
                    size="small"
                  />
                )}
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Created {String(c.created_at).slice(0, 19)}
              </Typography>
            </CardContent>
          </Card>
        )
      })}
    </Stack>
  )
}
