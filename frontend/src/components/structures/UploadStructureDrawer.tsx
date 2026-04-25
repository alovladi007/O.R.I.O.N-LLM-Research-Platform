// @ts-nocheck — MUI Drawer / TextField sx-prop union; pre-existing
// project pattern.
'use client'

/**
 * Phase 9 / Session 9.2 — upload-with-preview drawer.
 *
 * Drag-and-drop a CIF / POSCAR / XYZ → POST /structures/parse → render
 * the preview (lattice + spacegroup + atom count). User confirms →
 * POST /structures with the parsed payload → success toast +
 * redirect to detail page.
 *
 * Validation:
 *   - Reject files > 1 MB.
 *   - Reject non-``.cif | .poscar | .xyz | .vasp`` extensions.
 *   - Server 422s surface in the same toast pattern.
 */

import { useCallback, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Divider,
  Drawer,
  Stack,
  Typography,
} from '@mui/material'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'

import {
  api,
  formatErrorMessage,
  type StructureParseResponse,
} from '@/lib/api'

const MAX_BYTES = 1_000_000  // 1 MB
const ALLOWED_EXTS = new Set(['cif', 'poscar', 'xyz', 'vasp'])

function detectFormat(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  if (ext === 'cif') return 'CIF'
  if (ext === 'poscar' || ext === 'vasp') return 'POSCAR'
  if (ext === 'xyz') return 'XYZ'
  return ext.toUpperCase()
}

interface Props {
  open: boolean
  onClose: () => void
  /** Called with the new structure id after a successful save. */
  onSaved: (id: string) => void
}

export function UploadStructureDrawer({ open, onClose, onSaved }: Props) {
  const [parsing, setParsing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [preview, setPreview] = useState<StructureParseResponse | null>(null)
  const [previewText, setPreviewText] = useState<string>('')
  const [previewFormat, setPreviewFormat] = useState<string>('')
  const [error, setError] = useState<string>('')

  const reset = () => {
    setPreview(null)
    setPreviewText('')
    setPreviewFormat('')
    setError('')
  }

  const handleClose = () => {
    reset()
    onClose()
  }

  const onDrop = useCallback(async (files: File[]) => {
    setError('')
    if (!files.length) return
    const file = files[0]
    if (file.size > MAX_BYTES) {
      const msg = `File too large (${file.size} > ${MAX_BYTES} bytes)`
      setError(msg)
      toast.error(msg)
      return
    }
    const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
    if (!ALLOWED_EXTS.has(ext)) {
      const msg = `Unsupported extension '.${ext}'. Allowed: .cif .poscar .xyz .vasp`
      setError(msg)
      toast.error(msg)
      return
    }
    setParsing(true)
    try {
      const text = await file.text()
      const fmt = detectFormat(file.name)
      const parsed = await api.structures.parse({ text, format: fmt })
      setPreview(parsed)
      setPreviewText(text)
      setPreviewFormat(fmt)
    } catch (err) {
      const msg = formatErrorMessage(err)
      setError(msg)
      toast.error(msg)
    } finally {
      setParsing(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'chemical/x-cif': ['.cif'],
      'chemical/x-vasp': ['.poscar', '.vasp'],
      'chemical/x-xyz': ['.xyz'],
    },
  })

  const handleConfirm = async () => {
    if (!preview) return
    setSaving(true)
    try {
      const created = await api.structures.create({
        name: preview.formula,
        format: previewFormat,
        text: previewText,
        formula: preview.formula,
        num_atoms: preview.num_atoms,
        space_group: preview.space_group ?? null,
        space_group_number: preview.space_group_number ?? null,
      } as unknown as Parameters<typeof api.structures.create>[0])
      toast.success(`Saved structure ${created.id}`)
      onSaved(created.id as string)
    } catch (err) {
      const msg = formatErrorMessage(err)
      setError(msg)
      toast.error(msg)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={handleClose}
      PaperProps={{ sx: { width: { xs: '100%', md: 480 } } }}
    >
      <Box sx={{ p: 3 }} data-testid="upload-drawer">
        <Typography variant="h6" sx={{ mb: 2 }}>
          Upload structure
        </Typography>

        {!preview && (
          <Box
            {...getRootProps()}
            data-testid="upload-dropzone"
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'divider',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              bgcolor: isDragActive ? 'action.hover' : 'background.paper',
            }}
          >
            <input {...getInputProps()} data-testid="upload-input" />
            {parsing ? (
              <Stack alignItems="center" spacing={1}>
                <CircularProgress />
                <Typography variant="body2">Parsing…</Typography>
              </Stack>
            ) : (
              <>
                <Typography variant="body1">
                  {isDragActive
                    ? 'Drop here'
                    : 'Drag a CIF / POSCAR / XYZ here, or click to browse'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Max 1 MB
                </Typography>
              </>
            )}
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }} data-testid="upload-error">
            {error}
          </Alert>
        )}

        {preview && (
          <Stack spacing={2} sx={{ mt: 2 }} data-testid="upload-preview">
            <Typography variant="subtitle2">Preview</Typography>
            <Divider />
            <Field label="Formula" value={preview.formula} testId="preview-formula" />
            <Field
              label="Atoms (parsed cell)"
              value={`${preview.num_atoms}`}
              testId="preview-atoms"
            />
            <Field
              label="Spacegroup"
              value={
                preview.space_group_number != null
                  ? `${preview.space_group ?? '—'} (${preview.space_group_number})`
                  : '—'
              }
              testId="preview-spacegroup"
            />
            <Field
              label="a / b / c (Å)"
              value={
                preview.lattice_parameters
                  ? `${preview.lattice_parameters.a.toFixed(3)} / ` +
                    `${preview.lattice_parameters.b.toFixed(3)} / ` +
                    `${preview.lattice_parameters.c.toFixed(3)}`
                  : '—'
              }
              testId="preview-abc"
            />
            <Field
              label="α / β / γ (°)"
              value={
                preview.lattice_parameters
                  ? `${preview.lattice_parameters.alpha.toFixed(2)} / ` +
                    `${preview.lattice_parameters.beta.toFixed(2)} / ` +
                    `${preview.lattice_parameters.gamma.toFixed(2)}`
                  : '—'
              }
              testId="preview-angles"
            />

            <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
              <Button onClick={reset} disabled={saving}>
                Choose different
              </Button>
              <Button
                variant="contained"
                onClick={handleConfirm}
                disabled={saving}
                data-testid="upload-confirm"
              >
                {saving ? 'Saving…' : 'Save'}
              </Button>
            </Stack>
          </Stack>
        )}
      </Box>
    </Drawer>
  )
}

function Field({
  label,
  value,
  testId,
}: {
  label: string
  value: string
  testId: string
}) {
  return (
    <Stack direction="row" spacing={2}>
      <Typography variant="caption" sx={{ width: 120, color: 'text.secondary' }}>
        {label}
      </Typography>
      <Typography variant="body2" data-testid={testId}>
        {value}
      </Typography>
    </Stack>
  )
}
