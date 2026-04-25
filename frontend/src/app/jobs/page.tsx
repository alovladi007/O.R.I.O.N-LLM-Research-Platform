// @ts-nocheck — MUI X DataGrid + sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.3 — /jobs list page.
 *
 * DataGrid backed by ``api.jobs.list()`` with filters (status,
 * engine, kind, owner) and a "Refresh" button. Per-row actions:
 * Cancel (POST /jobs/{id}/cancel) and View detail (router.push).
 *
 * Live updates: a ``per-row`` SSE feed is overkill for a list view;
 * instead we open ONE stream against any *running* job we see and
 * update the cache when its state transitions. A 5-s react-query
 * refetch handles new arrivals (since Session 1.4's polling SSE on
 * the jobs router doesn't have a "list-level" event source). Phase
 * 10's Redis pub/sub will let us subscribe to a single
 * /jobs/events firehose; until then this hybrid keeps the grid
 * in sync without per-row polling.
 */

import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  Alert,
  Box,
  Button,
  Chip,
  IconButton,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
  Tooltip,
} from '@mui/material'
import { DataGrid, type GridColDef } from '@mui/x-data-grid'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Cancel, Refresh, Visibility, Replay } from '@mui/icons-material'
import toast from 'react-hot-toast'

import {
  api,
  formatErrorMessage,
  openSse,
  useRequireRole,
  type JobRow,
} from '@/lib/api'

const STATUS_COLORS: Record<string, 'default' | 'info' | 'warning' | 'success' | 'error'> = {
  PENDING: 'default',
  QUEUED: 'default',
  RUNNING: 'info',
  SUCCEEDED: 'success',
  COMPLETED: 'success',
  FAILED: 'error',
  CANCELLED: 'warning',
  TIMEOUT: 'warning',
}

const TERMINAL_STATES = new Set(['SUCCEEDED', 'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'])

const PAGE_SIZE_OPTIONS = [25, 50, 100] as const

interface FilterState {
  status?: string
  engine?: string
  kind?: string
}

export default function JobsListPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const router = useRouter()
  const qc = useQueryClient()
  const [filters, setFilters] = useState<FilterState>({})
  const [paginationModel, setPaginationModel] = useState({ page: 0, pageSize: 25 })

  const queryKey = useMemo(
    () => ['jobs', filters, paginationModel.page, paginationModel.pageSize],
    [filters, paginationModel],
  )

  const { data, isLoading, error, refetch } = useQuery({
    queryKey,
    queryFn: () =>
      api.jobs.list({
        ...filters,
        offset: paginationModel.page * paginationModel.pageSize,
        limit: paginationModel.pageSize,
      }),
    refetchInterval: 5_000,
    staleTime: 2_000,
  })

  // Open a per-row SSE for any *running* job in the current page so
  // the row updates immediately on terminal transition without
  // waiting for the 5 s refetch tick.
  useEffect(() => {
    const running = (data?.items ?? []).filter(
      (j) => !TERMINAL_STATES.has((j.status ?? '').toUpperCase()),
    )
    const handles = running.map((j) =>
      openSse(`/jobs/${j.id}/events`, {
        on: {
          status: (d: any) => {
            qc.setQueryData<typeof data>(queryKey, (prev) => {
              if (!prev) return prev
              return {
                ...prev,
                items: prev.items.map((row) =>
                  row.id === d.job_id
                    ? { ...row, status: d.status, progress: d.progress ?? row.progress }
                    : row,
                ),
              }
            })
          },
          terminal: () => refetch(),
        },
      }),
    )
    return () => handles.forEach((h) => h.cancel())
  }, [data?.items, qc, queryKey, refetch])

  const cancelJob = async (id: string) => {
    try {
      const updated = await api.jobs.cancel(id)
      toast.success(`Cancelled ${id.slice(0, 8)}`)
      qc.setQueryData<typeof data>(queryKey, (prev) =>
        prev
          ? { ...prev, items: prev.items.map((j) => (j.id === id ? updated : j)) }
          : prev,
      )
    } catch (err) {
      toast.error(formatErrorMessage(err))
    }
  }

  const reRunJob = async (job: JobRow) => {
    try {
      const created = await api.jobs.create({
        structure_id: job.structure_id,
        workflow_template_id: job.workflow_template_id,
        name: `${job.name ?? job.engine}-rerun`,
        engine: job.engine,
        parameters: job.parameters,
        priority: job.priority,
      })
      toast.success(`Re-submitted as ${(created.id as string).slice(0, 8)}`)
      router.push(`/jobs/${created.id}`)
    } catch (err) {
      toast.error(formatErrorMessage(err))
    }
  }

  const columns: GridColDef[] = [
    {
      field: 'status',
      headerName: 'Status',
      width: 110,
      renderCell: (params) => (
        <Chip
          label={params.value}
          color={STATUS_COLORS[(params.value as string)?.toUpperCase()] ?? 'default'}
          size="small"
        />
      ),
    },
    { field: 'name', headerName: 'Name', flex: 1, minWidth: 160 },
    { field: 'engine', headerName: 'Engine', width: 120 },
    {
      field: 'progress',
      headerName: 'Progress',
      width: 100,
      valueFormatter: (v: number | null) =>
        v == null ? '—' : `${Math.round(v * 100)}%`,
    },
    { field: 'submitted_at', headerName: 'Submitted', width: 180 },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 160,
      sortable: false,
      filterable: false,
      renderCell: (params) => {
        const job = params.row as JobRow
        const isTerminal = TERMINAL_STATES.has((job.status ?? '').toUpperCase())
        return (
          <Stack direction="row" spacing={0.5}>
            <Tooltip title="View detail">
              <IconButton
                size="small"
                data-testid={`job-view-${job.id}`}
                onClick={(e) => {
                  e.stopPropagation()
                  router.push(`/jobs/${job.id}`)
                }}
              >
                <Visibility fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title={isTerminal ? 'Already finished' : 'Cancel'}>
              <span>
                <IconButton
                  size="small"
                  disabled={isTerminal}
                  data-testid={`job-cancel-${job.id}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    void cancelJob(job.id as string)
                  }}
                >
                  <Cancel fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title="Re-run">
              <IconButton
                size="small"
                data-testid={`job-rerun-${job.id}`}
                onClick={(e) => {
                  e.stopPropagation()
                  void reRunJob(job)
                }}
              >
                <Replay fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        )
      },
    },
  ]

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, flexGrow: 1 }}>
          Jobs
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => refetch()}
          data-testid="jobs-refresh"
        >
          Refresh
        </Button>
      </Stack>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2 }}>
        <Select
          size="small"
          displayEmpty
          value={filters.status ?? ''}
          onChange={(e) =>
            setFilters({ ...filters, status: e.target.value || undefined })
          }
          sx={{ minWidth: 140 }}
          inputProps={{ 'data-testid': 'filter-status' }}
        >
          <MenuItem value="">All statuses</MenuItem>
          {Object.keys(STATUS_COLORS).map((s) => (
            <MenuItem key={s} value={s}>
              {s}
            </MenuItem>
          ))}
        </Select>
        <TextField
          size="small"
          label="Engine"
          value={filters.engine ?? ''}
          onChange={(e) =>
            setFilters({ ...filters, engine: e.target.value || undefined })
          }
          inputProps={{ 'data-testid': 'filter-engine' }}
        />
        <TextField
          size="small"
          label="Kind"
          value={filters.kind ?? ''}
          onChange={(e) =>
            setFilters({ ...filters, kind: e.target.value || undefined })
          }
          inputProps={{ 'data-testid': 'filter-kind' }}
        />
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {formatErrorMessage(error)}
        </Alert>
      )}

      <Box sx={{ height: 640 }}>
        <DataGrid
          rows={data?.items ?? []}
          columns={columns}
          getRowId={(r) => r.id}
          rowCount={data?.total ?? 0}
          loading={isLoading}
          pagination
          paginationMode="server"
          paginationModel={paginationModel}
          onPaginationModelChange={setPaginationModel}
          pageSizeOptions={PAGE_SIZE_OPTIONS as unknown as number[]}
          onRowClick={(p) => router.push(`/jobs/${p.id}`)}
          data-testid="jobs-grid"
        />
      </Box>
    </Box>
  )
}
