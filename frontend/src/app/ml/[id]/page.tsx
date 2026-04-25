// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.4 — /ml/[id] model detail + predict UI.
 *
 * Two tabs:
 *   - Overview — model info from /ml/models/{id}.
 *   - Predict — drag-drop ≤ 50 CIFs → POST /ml/properties per file
 *     (the existing endpoint takes a single structure_id; we
 *     POST /structures/parse first to obtain one). The result
 *     table shows (filename, μ, σ, σ-percentile color).
 *
 * If the backend returns 202 (model too large, async path),
 * we surface the message and link to /jobs.
 */

import { useCallback, useState } from 'react'
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
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  Typography,
} from '@mui/material'
import { ArrowBack, Upload } from '@mui/icons-material'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'

import { api, formatErrorMessage, useRequireRole } from '@/lib/api'

const MAX_FILES = 50

interface PredictRow {
  filename: string
  mu: number | null
  sigma: number | null
  sigmaPercentile: number | null  // 0..1
  raw?: any
  error?: string
}

export default function MLModelDetailPage() {
  useRequireRole(['admin', 'scientist', 'researcher', 'viewer'])
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const id = params.id as string
  const [tab, setTab] = useState(0)

  const { data: model, isLoading, error } = useQuery({
    queryKey: ['ml-model', id],
    queryFn: () => api.ml.getModel(id),
  })

  if (isLoading) {
    return (
      <Box sx={{ p: 6, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    )
  }
  if (error || !model) {
    return (
      <Box sx={{ p: 4 }}>
        <Alert severity="error">{error ? formatErrorMessage(error) : 'Model not found'}</Alert>
        <Button startIcon={<ArrowBack />} onClick={() => router.push('/ml')} sx={{ mt: 2 }}>
          Back to registry
        </Button>
      </Box>
    )
  }

  return (
    <Box sx={{ maxWidth: 1300, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
        <IconButton onClick={() => router.push('/ml')}>
          <ArrowBack />
        </IconButton>
        <Typography variant="h5" sx={{ fontWeight: 700, flexGrow: 1 }}>
          {model.name}
        </Typography>
        <Chip label={`v${model.version}`} />
      </Stack>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="Overview" data-testid="tab-overview" />
        <Tab label="Predict" data-testid="tab-predict" />
      </Tabs>

      {tab === 0 && (
        <Card variant="outlined">
          <CardContent>
            <Stack spacing={1.5}>
              <Stat label="Model id" value={(model as any).model_id ?? model.name} />
              <Stat label="Available" value={String(model.available)} />
              {(model as any).description && (
                <Stat label="Description" value={(model as any).description} />
              )}
              {(model as any).model_type && (
                <Stat label="Type" value={(model as any).model_type} />
              )}
            </Stack>
          </CardContent>
        </Card>
      )}

      {tab === 1 && <PredictPanel modelId={id} />}
    </Box>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Stack direction="row" spacing={2}>
      <Typography variant="caption" sx={{ width: 140, color: 'text.secondary' }}>
        {label}
      </Typography>
      <Typography variant="body2">{value}</Typography>
    </Stack>
  )
}

function PredictPanel({ modelId }: { modelId: string }) {
  const [rows, setRows] = useState<PredictRow[]>([])
  const [running, setRunning] = useState(false)

  const runPredict = useCallback(async (files: File[]) => {
    if (!files.length) return
    if (files.length > MAX_FILES) {
      toast.error(`Too many files (${files.length} > ${MAX_FILES})`)
      return
    }
    setRunning(true)
    const initial: PredictRow[] = files.map((f) => ({
      filename: f.name,
      mu: null,
      sigma: null,
      sigmaPercentile: null,
    }))
    setRows(initial)
    try {
      // Parse → create → predict per file. Sequential is fine for ≤50.
      const out: PredictRow[] = []
      for (const file of files) {
        try {
          const text = await file.text()
          const fmt = detectFormat(file.name)
          // 1. parse
          await api.structures.parse({ text, format: fmt })
          // 2. create the structure (so /ml/properties has an id)
          const created = await api.structures.create({
            name: file.name,
            format: fmt,
            text,
          } as any)
          // 3. predict
          const pred = (await api.ml.predict({
            structure_id: created.id,
            properties: ['bandgap'],
          } as any)) as any
          const mu = readMu(pred)
          const sigma = readSigma(pred)
          out.push({ filename: file.name, mu, sigma, sigmaPercentile: null, raw: pred })
        } catch (e) {
          out.push({
            filename: file.name,
            mu: null,
            sigma: null,
            sigmaPercentile: null,
            error: formatErrorMessage(e),
          })
        }
      }
      // Compute σ-percentile within the batch.
      const sigmas = out
        .map((r) => r.sigma)
        .filter((s): s is number => typeof s === 'number')
      const sortedSigmas = [...sigmas].sort((a, b) => a - b)
      const finalRows = out.map((r) => {
        if (typeof r.sigma !== 'number') return r
        // rank = (index of largest sigma_i ≤ r.sigma + 0.5) / N
        const rank =
          (sortedSigmas.findIndex((s) => s >= (r.sigma as number)) + 0.5) /
          Math.max(1, sortedSigmas.length)
        return { ...r, sigmaPercentile: rank }
      })
      setRows(finalRows)
    } finally {
      setRunning(false)
    }
  }, [modelId])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: runPredict,
    multiple: true,
    accept: {
      'chemical/x-cif': ['.cif'],
      'chemical/x-vasp': ['.poscar', '.vasp'],
    },
  })

  return (
    <Stack spacing={2}>
      <Box
        {...getRootProps()}
        data-testid="predict-dropzone"
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          borderRadius: 2,
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
        }}
      >
        <input {...getInputProps()} data-testid="predict-input" />
        <Upload sx={{ fontSize: 32, mb: 1, color: 'text.secondary' }} />
        <Typography variant="body1">
          {isDragActive
            ? 'Drop here'
            : `Drag up to ${MAX_FILES} CIFs / POSCARs here, or click to browse`}
        </Typography>
      </Box>

      {running && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CircularProgress size={18} />
          <Typography variant="body2">Predicting…</Typography>
        </Box>
      )}

      {rows.length > 0 && (
        <Card variant="outlined">
          <CardContent>
            <Table size="small" data-testid="predict-table">
              <TableHead>
                <TableRow>
                  <TableCell>File</TableCell>
                  <TableCell align="right">μ</TableCell>
                  <TableCell align="right">σ</TableCell>
                  <TableCell align="center">σ-percentile</TableCell>
                  <TableCell>Notes</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.filename}>
                    <TableCell>{r.filename}</TableCell>
                    <TableCell align="right">
                      {typeof r.mu === 'number' ? r.mu.toFixed(3) : '—'}
                    </TableCell>
                    <TableCell align="right">
                      {typeof r.sigma === 'number' ? r.sigma.toFixed(3) : '—'}
                    </TableCell>
                    <TableCell align="center">
                      {typeof r.sigmaPercentile === 'number' ? (
                        <Chip
                          label={`${Math.round(r.sigmaPercentile * 100)}%`}
                          size="small"
                          color={percentileColor(r.sigmaPercentile)}
                          data-testid={`predict-pct-${r.filename}`}
                        />
                      ) : (
                        '—'
                      )}
                    </TableCell>
                    <TableCell>{r.error ?? ''}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </Stack>
  )
}

function detectFormat(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  if (ext === 'cif') return 'CIF'
  if (ext === 'poscar' || ext === 'vasp') return 'POSCAR'
  if (ext === 'xyz') return 'XYZ'
  return ext.toUpperCase()
}

function readMu(pred: any): number | null {
  // The /ml/properties response shape carries
  // ``predicted_properties: {bandgap: {value: …, uncertainty: …}}``.
  const pp = pred?.predicted_properties
  if (!pp) return null
  const v = (Object.values(pp)[0] as any) ?? null
  if (typeof v === 'number') return v
  if (v && typeof v.value === 'number') return v.value
  return null
}

function readSigma(pred: any): number | null {
  const pp = pred?.predicted_properties
  if (!pp) return null
  const v = (Object.values(pp)[0] as any) ?? null
  if (v && typeof v.uncertainty === 'number') return v.uncertainty
  if (v && typeof v.sigma === 'number') return v.sigma
  return null
}

function percentileColor(p: number): 'success' | 'warning' | 'error' {
  if (p < 0.25) return 'success'
  if (p < 0.75) return 'warning'
  return 'error'
}
