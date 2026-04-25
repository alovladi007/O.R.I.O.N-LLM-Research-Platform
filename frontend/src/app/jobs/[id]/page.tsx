// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.3 — /jobs/[id] detail.
 *
 * Tabs: Inputs · Outputs · Logs (LogTail SSE) · Artifacts · Provenance.
 * Header: status chip, engine, progress %, action buttons (Cancel,
 * Re-run, Open workflow if linked).
 */

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useQuery, useQueryClient } from '@tanstack/react-query'
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
import { ArrowBack, Cancel, Replay } from '@mui/icons-material'
import toast from 'react-hot-toast'

import {
  api,
  formatErrorMessage,
  openSse,
  useRequireRole,
} from '@/lib/api'
import { LogTail } from '@/components/jobs/LogTail'

const TERMINAL_STATES = new Set(['SUCCEEDED', 'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'])

export default function JobDetailPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const qc = useQueryClient()
  const id = params.id as string
  const [tab, setTab] = useState(0)

  const { data: job, isLoading, error, refetch } = useQuery({
    queryKey: ['job', id],
    queryFn: () => api.jobs.get(id),
    refetchInterval: (q) => {
      const j = q.state.data
      if (!j || TERMINAL_STATES.has((j.status ?? '').toUpperCase())) return false
      return 5_000
    },
  })

  // SSE → react-query cache update so the header repaints live.
  useEffect(() => {
    if (!id) return
    const handle = openSse(`/jobs/${id}/events`, {
      on: {
        status: (d: any) => {
          qc.setQueryData(['job', id], (prev: any) =>
            prev ? { ...prev, status: d.status, progress: d.progress, current_step: d.current_step } : prev,
          )
        },
        terminal: () => refetch(),
      },
    })
    return () => handle.cancel()
  }, [id, qc, refetch])

  const cancelJob = async () => {
    try {
      await api.jobs.cancel(id)
      toast.success('Cancelled')
      refetch()
    } catch (err) {
      toast.error(formatErrorMessage(err))
    }
  }

  const reRun = async () => {
    if (!job) return
    try {
      const created = await api.jobs.create({
        structure_id: job.structure_id,
        workflow_template_id: job.workflow_template_id,
        engine: job.engine,
        parameters: job.parameters,
        priority: job.priority,
      })
      router.push(`/jobs/${created.id}`)
    } catch (err) {
      toast.error(formatErrorMessage(err))
    }
  }

  if (isLoading) {
    return (
      <Box sx={{ p: 6, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    )
  }
  if (error || !job) {
    return (
      <Box sx={{ p: 4 }}>
        <Alert severity="error">{error ? formatErrorMessage(error) : 'Job not found'}</Alert>
        <Button startIcon={<ArrowBack />} onClick={() => router.push('/jobs')} sx={{ mt: 2 }}>
          Back to jobs
        </Button>
      </Box>
    )
  }

  const isTerminal = TERMINAL_STATES.has((job.status ?? '').toUpperCase())

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
        <IconButton onClick={() => router.push('/jobs')}>
          <ArrowBack />
        </IconButton>
        <Typography variant="h5" sx={{ fontWeight: 700, flexGrow: 1 }}>
          {job.name ?? job.engine ?? id}
        </Typography>
        <Chip label={job.status} data-testid="job-status-chip" />
        {!isTerminal && (
          <Button
            variant="outlined"
            color="warning"
            startIcon={<Cancel />}
            onClick={cancelJob}
            data-testid="job-detail-cancel"
          >
            Cancel
          </Button>
        )}
        <Button variant="outlined" startIcon={<Replay />} onClick={reRun}>
          Re-run
        </Button>
      </Stack>

      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction="row" spacing={3} flexWrap="wrap" useFlexGap>
            <Stat label="Engine" value={job.engine ?? '—'} />
            <Stat label="Step" value={job.current_step ?? '—'} />
            <Stat
              label="Progress"
              value={
                job.progress != null ? `${Math.round(job.progress * 100)}%` : '—'
              }
            />
            <Stat label="Submitted" value={String(job.submitted_at ?? '—')} />
            <Stat label="Started" value={String(job.started_at ?? '—')} />
            <Stat label="Finished" value={String(job.finished_at ?? '—')} />
            {job.error_message && (
              <Stat label="Error" value={job.error_message} testId="job-error" />
            )}
          </Stack>
        </CardContent>
      </Card>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="Inputs" data-testid="tab-inputs" />
        <Tab label="Outputs" data-testid="tab-outputs" />
        <Tab label="Logs" data-testid="tab-logs" />
        <Tab label="Artifacts" data-testid="tab-artifacts" />
        <Tab label="Provenance" data-testid="tab-provenance" />
      </Tabs>

      {tab === 0 && (
        <JsonBlock value={job.parameters} testId="job-inputs-json" />
      )}
      {tab === 1 && (
        <JsonBlock
          value={(job as any).outputs ?? null}
          empty="Outputs will appear once the job completes."
          testId="job-outputs-json"
        />
      )}
      {tab === 2 && <LogTail jobId={id} />}
      {tab === 3 && (
        <Alert severity="info">
          Artifact downloads (presigned MinIO URLs) wire in alongside the
          Phase 2.1 artifact bundler. Job extra_metadata.artifact may
          carry a path: <code>{(job as any).extra_metadata?.artifact ?? '—'}</code>
        </Alert>
      )}
      {tab === 4 && (
        <Alert severity="info">
          Provenance graph lands with Phase 12. Job created at{' '}
          {String(job.submitted_at)}; structure_id ={' '}
          <code>{job.structure_id}</code>.
        </Alert>
      )}
    </Box>
  )
}

function Stat({
  label,
  value,
  testId,
}: {
  label: string
  value: string
  testId?: string
}) {
  return (
    <Box sx={{ minWidth: 120 }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" fontWeight={600} data-testid={testId}>
        {value}
      </Typography>
    </Box>
  )
}

function JsonBlock({
  value,
  empty,
  testId,
}: {
  value: unknown
  empty?: string
  testId?: string
}) {
  if (value == null) {
    return <Alert severity="info">{empty ?? 'No data.'}</Alert>
  }
  const text = JSON.stringify(value, null, 2)
  return (
    <Box
      component="pre"
      data-testid={testId}
      sx={{
        bgcolor: '#0a0a14',
        color: '#cccccc',
        p: 2,
        borderRadius: 1,
        overflow: 'auto',
        fontSize: 13,
        maxHeight: 600,
      }}
    >
      {text}
    </Box>
  )
}
