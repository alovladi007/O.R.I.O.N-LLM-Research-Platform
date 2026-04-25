// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.4 — /campaigns/[id] detail.
 *
 * Header, best-so-far line plot (recharts), cycles table.
 * AL campaigns are returned in-memory and synchronous today, so
 * "auto-refresh" is mostly a no-op (the backend never updates them
 * after the create call); we still poll every 5 s so a 7.2b-promoted
 * persistent campaign can update without a manual refresh.
 */

import { useMemo } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material'
import { ArrowBack } from '@mui/icons-material'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { api, formatErrorMessage, useRequireRole } from '@/lib/api'

export default function CampaignDetailPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const id = params.id as string

  const { data: campaign, isLoading, error } = useQuery({
    queryKey: ['al-campaign', id],
    queryFn: () => api.al.get(id),
    refetchInterval: 5_000,
  })

  const bestSoFar = useMemo(() => {
    if (!campaign) return [] as { cycle: number; best: number }[]
    return (campaign.cumulative_best_history ?? []).map((b: number, i: number) => ({
      cycle: i,
      best: b,
    }))
  }, [campaign])

  if (isLoading) {
    return (
      <Box sx={{ p: 6, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    )
  }
  if (error || !campaign) {
    return (
      <Box sx={{ p: 4 }}>
        <Alert severity="error">
          {error ? formatErrorMessage(error) : 'Campaign not found'}
        </Alert>
        <Button startIcon={<ArrowBack />} onClick={() => router.push('/campaigns')} sx={{ mt: 2 }}>
          Back
        </Button>
      </Box>
    )
  }

  return (
    <Box sx={{ maxWidth: 1300, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
        <IconButton onClick={() => router.push('/campaigns')}>
          <ArrowBack />
        </IconButton>
        <Typography variant="h5" sx={{ fontWeight: 700, flexGrow: 1 }}>
          {campaign.name}
        </Typography>
        <Chip
          label={`${campaign.n_cycles_completed} cycles`}
          size="small"
        />
        {campaign.final_val_mae != null && (
          <Chip
            label={`val MAE ${campaign.final_val_mae.toFixed(3)}`}
            color="success"
            size="small"
            data-testid="campaign-final-mae"
          />
        )}
      </Stack>

      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Best-so-far
          </Typography>
          <Box sx={{ height: 280 }} data-testid="best-so-far-chart">
            <ResponsiveContainer>
              <LineChart data={bestSoFar} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="cycle" label={{ value: 'cycle', position: 'insideBottom', offset: -2 }} />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="best"
                  stroke="#1e3a8a"
                  strokeWidth={2}
                  dot
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Cycles
          </Typography>
          <Table size="small" data-testid="cycles-table">
            <TableHead>
              <TableRow>
                <TableCell>#</TableCell>
                <TableCell>Queried</TableCell>
                <TableCell>Targets</TableCell>
                <TableCell align="right">val MAE</TableCell>
                <TableCell align="right">Best</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {(campaign.cycles ?? []).map((c: any) => (
                <TableRow key={c.cycle_index}>
                  <TableCell>{c.cycle_index}</TableCell>
                  <TableCell>
                    {(c.queried_indices ?? []).join(', ')}
                  </TableCell>
                  <TableCell>
                    {(c.queried_targets ?? [])
                      .map((v: number) => v.toFixed(3))
                      .join(', ')}
                  </TableCell>
                  <TableCell align="right">
                    {c.val_mae != null ? c.val_mae.toFixed(4) : '—'}
                  </TableCell>
                  <TableCell align="right">
                    {c.cumulative_best != null
                      ? c.cumulative_best.toFixed(3)
                      : '—'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Box>
  )
}
