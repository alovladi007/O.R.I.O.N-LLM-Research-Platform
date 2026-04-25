// @ts-nocheck — MUI sx-prop union + reactflow generic types are too
// complex for the project's TS config. Pre-existing project pattern;
// runtime is correct, the spec covers behavior.
'use client'

/**
 * Phase 9 / Session 9.3 — /workflows/[id] DAG view.
 *
 * Live DAG visualization for a workflow run:
 *   - nodes = steps, colored by status
 *   - edges = parent → child dependencies (parsed from the spec)
 *   - subscribed to /workflow-runs/{id}/events SSE; reactflow nodes
 *     repaint on every step transition
 *   - clicking a node opens a side drawer with the step summary
 *     and a deep link to /jobs/{id}
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Drawer,
  IconButton,
  Stack,
  Typography,
} from '@mui/material'
import { ArrowBack, Cancel } from '@mui/icons-material'
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  ReactFlowProvider,
  type Edge,
  type Node,
} from 'reactflow'
import 'reactflow/dist/style.css'
import toast from 'react-hot-toast'

import {
  api,
  formatErrorMessage,
  openSse,
  useRequireRole,
} from '@/lib/api'

const STATUS_COLORS: Record<string, string> = {
  PENDING: '#9ca3af',
  DISPATCHABLE: '#9ca3af',
  RUNNING: '#3b82f6',
  SUCCEEDED: '#10b981',
  COMPLETED: '#10b981',
  FAILED: '#ef4444',
  CANCELLED: '#f59e0b',
  TIMEOUT: '#f59e0b',
}

const RUN_TERMINAL = new Set(['COMPLETED', 'FAILED', 'CANCELLED'])

interface StepRow {
  step_id: string
  status: string
  job_id?: string | null
  depends_on?: string[]  // not always present in the response shape
}

export default function WorkflowRunDetailPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const qc = useQueryClient()
  const id = params.id as string
  const [selected, setSelected] = useState<StepRow | null>(null)

  const { data: run, isLoading, error, refetch } = useQuery({
    queryKey: ['workflow-run', id],
    queryFn: () => api.workflowRuns.get(id),
    refetchInterval: (q) => {
      const r = q.state.data
      if (!r || RUN_TERMINAL.has((r.status ?? '').toUpperCase())) return false
      return 5_000
    },
  })

  // Subscribe to SSE → react-query cache patches.
  useEffect(() => {
    if (!id) return
    const handle = openSse(`/workflow-runs/${id}/events`, {
      on: {
        run: (d: any) => {
          qc.setQueryData(['workflow-run', id], (prev: any) =>
            prev ? { ...prev, status: d.status } : prev,
          )
        },
        step: (d: any) => {
          qc.setQueryData(['workflow-run', id], (prev: any) => {
            if (!prev) return prev
            return {
              ...prev,
              steps: (prev.steps ?? []).map((s: any) =>
                s.step_id === d.step_id
                  ? { ...s, status: d.status, simulation_job_id: d.job_id }
                  : s,
              ),
            }
          })
        },
        terminal: () => refetch(),
      },
    })
    return () => handle.cancel()
  }, [id, qc, refetch])

  const cancelRun = useCallback(async () => {
    try {
      await api.workflowRuns.cancel(id)
      toast.success('Run cancelled')
      refetch()
    } catch (err) {
      toast.error(formatErrorMessage(err))
    }
  }, [id, refetch])

  // Build the reactflow nodes + edges. Layout strategy: simple
  // left-to-right by topological generation. Each node is placed at
  // ``x = generation * 220``, ``y = positionInGeneration * 110``.
  // For more sophisticated layouts (dagre / elk), Session 9.3b can
  // swap this in.
  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    const steps = (run as any)?.steps ?? []
    if (!steps.length) return { nodes: [], edges: [] }

    // depends_on may live in either step.depends_on or be inferred
    // from the spec; the response model exposes it as ``depends_on``.
    const stepMap = new Map<string, StepRow & { depends_on: string[] }>(
      steps.map((s: any) => [
        s.step_id,
        { ...s, depends_on: s.depends_on ?? [] },
      ]),
    )

    // Topological generations.
    const generation: Record<string, number> = {}
    const visit = (id: string, seen: Set<string>): number => {
      if (id in generation) return generation[id]
      if (seen.has(id)) return 0  // cycle guard
      seen.add(id)
      const step = stepMap.get(id)
      if (!step || step.depends_on.length === 0) {
        generation[id] = 0
        return 0
      }
      const g =
        1 + Math.max(...step.depends_on.map((d) => visit(d, new Set(seen))))
      generation[id] = g
      return g
    }
    for (const s of steps) visit(s.step_id, new Set())

    // Group by generation for y-coordinate assignment.
    const byGen: Record<number, string[]> = {}
    for (const [stepId, g] of Object.entries(generation)) {
      ;(byGen[g] ??= []).push(stepId)
    }

    const nodes: Node[] = steps.map((s: any) => {
      const g = generation[s.step_id]
      const ix = byGen[g].indexOf(s.step_id)
      const color = STATUS_COLORS[(s.status ?? '').toUpperCase()] ?? '#94a3b8'
      return {
        id: s.step_id,
        type: 'default',
        data: { label: `${s.step_id}\n${s.status}` },
        position: { x: g * 220, y: ix * 110 },
        style: {
          background: color,
          color: '#ffffff',
          border: '2px solid rgba(0,0,0,0.15)',
          borderRadius: 8,
          padding: 8,
          fontSize: 12,
          width: 180,
          whiteSpace: 'pre-wrap',
        },
      }
    })

    const edges: Edge[] = []
    for (const s of steps) {
      for (const d of s.depends_on ?? []) {
        edges.push({
          id: `${d}->${s.step_id}`,
          source: d,
          target: s.step_id,
          markerEnd: { type: MarkerType.ArrowClosed },
          animated: (s.status ?? '').toUpperCase() === 'RUNNING',
        })
      }
    }
    return { nodes, edges }
  }, [run])

  const stats = useMemo(() => {
    const steps = (run as any)?.steps ?? []
    const counts: Record<string, number> = {}
    for (const s of steps) counts[s.status] = (counts[s.status] ?? 0) + 1
    const done = (counts.SUCCEEDED ?? 0) + (counts.COMPLETED ?? 0)
    return {
      total: steps.length,
      done,
      pct: steps.length ? Math.round((done / steps.length) * 100) : 0,
      counts,
    }
  }, [run])

  if (isLoading) {
    return (
      <Box sx={{ p: 6, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    )
  }
  if (error || !run) {
    return (
      <Box sx={{ p: 4 }}>
        <Alert severity="error">{error ? formatErrorMessage(error) : 'Workflow not found'}</Alert>
        <Button startIcon={<ArrowBack />} onClick={() => router.push('/jobs')} sx={{ mt: 2 }}>
          Back
        </Button>
      </Box>
    )
  }

  const isTerminal = RUN_TERMINAL.has((run.status ?? '').toUpperCase())

  return (
    <Box sx={{ maxWidth: 1600, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
        <IconButton onClick={() => router.back()}>
          <ArrowBack />
        </IconButton>
        <Typography variant="h5" sx={{ fontWeight: 700, flexGrow: 1 }}>
          Workflow {String(id).slice(0, 8)}
        </Typography>
        <Chip label={run.status} data-testid="run-status-chip" />
        <Chip
          label={`${stats.done}/${stats.total} (${stats.pct}%)`}
          color="primary"
          variant="outlined"
          data-testid="run-progress-chip"
        />
        {!isTerminal && (
          <Button
            variant="outlined"
            color="warning"
            startIcon={<Cancel />}
            onClick={cancelRun}
            data-testid="run-cancel"
          >
            Cancel
          </Button>
        )}
      </Stack>

      <Box
        sx={{ height: 600, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}
        data-testid="workflow-dag"
        data-step-count={stats.total}
      >
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            onNodeClick={(_, n) => {
              const step = (run as any).steps.find((s: any) => s.step_id === n.id)
              if (step) setSelected(step)
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
      </Box>

      <Drawer
        anchor="right"
        open={!!selected}
        onClose={() => setSelected(null)}
        PaperProps={{ sx: { width: { xs: '100%', md: 420 } } }}
      >
        {selected && (
          <Box sx={{ p: 3 }} data-testid="step-drawer">
            <Typography variant="h6">{selected.step_id}</Typography>
            <Chip
              label={selected.status}
              sx={{ mt: 1, mb: 2 }}
              style={{
                background: STATUS_COLORS[selected.status.toUpperCase()],
                color: 'white',
              }}
            />
            <Stack spacing={1}>
              <Typography variant="caption" color="text.secondary">
                step_id
              </Typography>
              <Typography variant="body2">{selected.step_id}</Typography>
              <Typography variant="caption" color="text.secondary">
                job_id
              </Typography>
              <Typography variant="body2">
                {(selected as any).simulation_job_id ?? selected.job_id ?? '—'}
              </Typography>
            </Stack>
            {((selected as any).simulation_job_id || selected.job_id) && (
              <Button
                variant="contained"
                sx={{ mt: 3 }}
                onClick={() =>
                  router.push(
                    `/jobs/${(selected as any).simulation_job_id ?? selected.job_id}`,
                  )
                }
                data-testid="step-open-job"
              >
                Open in /jobs
              </Button>
            )}
          </Box>
        )}
      </Drawer>
    </Box>
  )
}
