// @ts-nocheck — MUI sx-prop union (project pattern).
'use client'

/**
 * Phase 9 / Session 9.4 — AL campaign create dialog.
 *
 * Form fields mirror :class:`backend.common.ml.active_learning_v2`'s
 * ALCampaignCreate exactly. The X_pool / y_pool inputs accept either
 * a CSV paste OR a file upload; we parse client-side into the
 * nested arrays the backend expects.
 *
 * The roadmap calls out a tabbed AL/BO selector here too; BO form
 * is parked behind 7.2b (no DB-backed campaigns_v2 endpoint yet),
 * so this dialog is AL-only.
 */

import { useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import toast from 'react-hot-toast'

import { api, formatErrorMessage, type ALCampaignCreate } from '@/lib/api'

interface Props {
  open: boolean
  onClose: () => void
  onCreated: (id: string) => void
}

const ACQUISITIONS = ['max_sigma', 'ucb', 'ei', 'bald'] as const
const MODEL_KINDS = ['mean', 'random_forest'] as const

export function CreateALCampaignDialog({ open, onClose, onCreated }: Props) {
  const [name, setName] = useState('')
  const [csvText, setCsvText] = useState('')
  const [seedIndices, setSeedIndices] = useState('0,1')
  const [acquisition, setAcquisition] =
    useState<typeof ACQUISITIONS[number]>('max_sigma')
  const [querySize, setQuerySize] = useState(2)
  const [nCycles, setNCycles] = useState(3)
  const [modelKind, setModelKind] =
    useState<typeof MODEL_KINDS[number]>('random_forest')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string>('')

  const reset = () => {
    setName('')
    setCsvText('')
    setSeedIndices('0,1')
    setAcquisition('max_sigma')
    setQuerySize(2)
    setNCycles(3)
    setModelKind('random_forest')
    setError('')
  }

  const submit = async () => {
    setError('')
    let X_pool: number[][]
    let y_pool: number[]
    try {
      const parsed = parseCsv(csvText)
      X_pool = parsed.X_pool
      y_pool = parsed.y_pool
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Invalid CSV'
      setError(m)
      return
    }
    const seeds = seedIndices
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n))
    const body: ALCampaignCreate = {
      name,
      X_pool,
      y_pool,
      initial_train_indices: seeds,
      acquisition,
      query_size: querySize,
      n_cycles: nCycles,
      maximize: false,
      beta: 2.0,
      xi: 0.0,
      seed: 0,
      model_kind: modelKind,
    } as unknown as ALCampaignCreate
    setSubmitting(true)
    try {
      const created = await api.al.create(body)
      toast.success(`Campaign created (${created.n_cycles_completed} cycles)`)
      onCreated(created.id as string)
      reset()
    } catch (err) {
      const msg = formatErrorMessage(err)
      setError(msg)
      toast.error(msg)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle data-testid="create-al-title">New AL campaign</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <TextField
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            inputProps={{ 'data-testid': 'al-name' }}
            fullWidth
            required
          />
          <TextField
            label="Pool (CSV: features…,target — one row per candidate)"
            value={csvText}
            onChange={(e) => setCsvText(e.target.value)}
            inputProps={{ 'data-testid': 'al-csv' }}
            multiline
            minRows={5}
            placeholder={`# example: 2 features + target
1.0, 2.0, 3.0
4.0, 5.0, 6.0
7.0, 8.0, 9.0`}
            fullWidth
            required
          />
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
            <TextField
              label="Initial train indices (comma-separated)"
              value={seedIndices}
              onChange={(e) => setSeedIndices(e.target.value)}
              inputProps={{ 'data-testid': 'al-seeds' }}
              fullWidth
            />
            <TextField
              select
              label="Acquisition"
              value={acquisition}
              onChange={(e) =>
                setAcquisition(e.target.value as typeof ACQUISITIONS[number])
              }
              inputProps={{ 'data-testid': 'al-acq' }}
              sx={{ minWidth: 160 }}
            >
              {ACQUISITIONS.map((a) => (
                <MenuItem key={a} value={a}>
                  {a}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              select
              label="Model"
              value={modelKind}
              onChange={(e) =>
                setModelKind(e.target.value as typeof MODEL_KINDS[number])
              }
              inputProps={{ 'data-testid': 'al-model' }}
              sx={{ minWidth: 160 }}
            >
              {MODEL_KINDS.map((m) => (
                <MenuItem key={m} value={m}>
                  {m}
                </MenuItem>
              ))}
            </TextField>
          </Stack>
          <Stack direction="row" spacing={2}>
            <TextField
              label="Query size"
              type="number"
              value={querySize}
              onChange={(e) => setQuerySize(parseInt(e.target.value, 10) || 1)}
              inputProps={{ 'data-testid': 'al-qs', min: 1, max: 200 }}
              sx={{ width: 140 }}
            />
            <TextField
              label="Cycles"
              type="number"
              value={nCycles}
              onChange={(e) => setNCycles(parseInt(e.target.value, 10) || 1)}
              inputProps={{ 'data-testid': 'al-cycles', min: 1, max: 50 }}
              sx={{ width: 140 }}
            />
          </Stack>
          {error && (
            <Alert severity="error" data-testid="al-create-error">
              {error}
            </Alert>
          )}
          <Typography variant="caption" color="text.secondary">
            The AL engine is synchronous (Session 6.5 in-memory store);
            larger corpora may take a few seconds to return. Cycles
            store in-process and survive the page reload until the
            backend restarts. Session 6.5b promotes to a DB-backed
            store + Celery dispatch.
          </Typography>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          disabled={submitting || !name || !csvText}
          onClick={submit}
          data-testid="al-create-submit"
        >
          {submitting ? 'Running…' : 'Create + run'}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

function parseCsv(text: string): { X_pool: number[][]; y_pool: number[] } {
  const rows = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
  if (rows.length < 2) {
    throw new Error('CSV must contain at least 2 non-comment rows')
  }
  const parsed = rows.map((line) =>
    line.split(/[,\s]+/).map((v) => parseFloat(v.trim())),
  )
  if (parsed.some((r) => r.some((v) => isNaN(v)))) {
    throw new Error('CSV contains a non-numeric value')
  }
  const cols = parsed[0].length
  if (parsed.some((r) => r.length !== cols)) {
    throw new Error('CSV row lengths are inconsistent')
  }
  if (cols < 2) {
    throw new Error('CSV needs at least one feature column + one target column')
  }
  const X_pool = parsed.map((r) => r.slice(0, cols - 1))
  const y_pool = parsed.map((r) => r[cols - 1])
  return { X_pool, y_pool }
}
