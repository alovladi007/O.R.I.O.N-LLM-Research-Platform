// @ts-nocheck — MUI sx-prop union complexity (project-wide pattern,
// see Session 9.1 dashboard / AppBar). Runtime is correct.
'use client'

/**
 * Phase 9 / Session 9.2 — /structures/[id] detail.
 *
 * Header: formula, name, spacegroup, density, # atoms, volume.
 * Tabs:
 *   - 3D viewer (StructureViewer; CPK + bonds + supercell + screenshot)
 *   - Lattice / sites (POSCAR-style table)
 *   - Properties (grouped by method+conditions; placeholder until
 *     /structures/{id}/properties wires in)
 *   - Provenance (placeholder until Phase 12)
 * Export: format selector → GET /structures/{id}/export?format=…
 */

import { useMemo, useState } from 'react'
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
  MenuItem,
  Select,
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
import { ArrowBack, Download } from '@mui/icons-material'
import { saveAs } from 'file-saver'
import toast from 'react-hot-toast'

import {
  api,
  formatErrorMessage,
  type StructureRow,
  useRequireRole,
} from '@/lib/api'
import StructureViewer from '@/components/structures/StructureViewer'

type ExportFormat = 'cif' | 'poscar' | 'xyz' | 'json'

export default function StructureDetailPage() {
  useRequireRole(['admin', 'scientist', 'viewer', 'researcher'])
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const id = params.id as string

  const [tab, setTab] = useState(0)
  const [exportFmt, setExportFmt] = useState<ExportFormat>('cif')

  const {
    data: structure,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['structure', id],
    queryFn: () => api.structures.get(id),
    staleTime: 60_000,
  })

  const viewerStructure = useMemo(() => buildViewerInput(structure), [structure])

  const handleExport = async () => {
    try {
      const blob = await api.structures.exportFile(id, exportFmt)
      const ext = exportFmt === 'poscar' ? 'POSCAR' : exportFmt
      saveAs(blob, `${structure?.formula ?? id}.${ext}`)
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
  if (error || !structure) {
    return (
      <Box sx={{ p: 4 }}>
        <Alert severity="error">
          {error ? formatErrorMessage(error) : 'Structure not found'}
        </Alert>
        <Button startIcon={<ArrowBack />} onClick={() => router.push('/structures')} sx={{ mt: 2 }}>
          Back to list
        </Button>
      </Box>
    )
  }

  return (
    <Box sx={{ maxWidth: 1400, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
        <IconButton onClick={() => router.push('/structures')}>
          <ArrowBack />
        </IconButton>
        <Typography variant="h4" sx={{ fontWeight: 700, flexGrow: 1 }}>
          {structure.formula ?? structure.name ?? id}
        </Typography>
        <Select
          size="small"
          value={exportFmt}
          onChange={(e) => setExportFmt(e.target.value as ExportFormat)}
          data-testid="export-format-select"
        >
          <MenuItem value="cif">CIF</MenuItem>
          <MenuItem value="poscar">POSCAR</MenuItem>
          <MenuItem value="xyz">XYZ</MenuItem>
          <MenuItem value="json">JSON</MenuItem>
        </Select>
        <Button
          variant="contained"
          startIcon={<Download />}
          onClick={handleExport}
          data-testid="export-button"
        >
          Export
        </Button>
      </Stack>

      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            <Stat label="Name" value={structure.name ?? '—'} />
            <Stat
              label="Spacegroup"
              value={
                structure.space_group_number != null
                  ? `${structure.space_group ?? '—'} (${structure.space_group_number})`
                  : structure.space_group ?? '—'
              }
            />
            <Stat label="Atoms" value={`${structure.num_atoms ?? '—'}`} testId="detail-num-atoms" />
            <Stat
              label="Density"
              value={
                structure.density != null
                  ? `${structure.density.toFixed(3)} g/cm³`
                  : '—'
              }
            />
            <Stat
              label="Volume"
              value={
                structure.volume != null
                  ? `${structure.volume.toFixed(2)} Å³`
                  : '—'
              }
            />
            <Stat
              label="Lattice (a/b/c)"
              value={formatABC(structure)}
              testId="detail-abc"
            />
            <Chip label={structure.format ?? '?'} size="small" />
          </Stack>
        </CardContent>
      </Card>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="3D Viewer" data-testid="tab-viewer" />
        <Tab label="Lattice & Sites" data-testid="tab-lattice" />
        <Tab label="Properties" data-testid="tab-properties" />
        <Tab label="Provenance" data-testid="tab-provenance" />
      </Tabs>

      {tab === 0 && (
        <Box>
          {viewerStructure ? (
            <StructureViewer
              structure={viewerStructure}
              testId="detail-viewer"
              height={520}
            />
          ) : (
            <Alert severity="info">
              No lattice / atoms data available for this structure (the
              detail endpoint did not return parsed coordinates). Re-upload
              the source file to populate.
            </Alert>
          )}
        </Box>
      )}

      {tab === 1 && <LatticeSitesTab structure={structure} />}
      {tab === 2 && (
        <Alert severity="info">
          Per-property table (grouped by method × conditions) wires in
          alongside the Phase 1.3 properties endpoint. Phase 8 elastic /
          phonon / defect properties surface here once persisted.
        </Alert>
      )}
      {tab === 3 && (
        <Alert severity="info">
          Provenance graph lands with Phase 12. For now: structure id{' '}
          <code>{id}</code> created at {structure.created_at ?? '—'}.
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
    <Box sx={{ minWidth: 110 }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" fontWeight={600} data-testid={testId}>
        {value}
      </Typography>
    </Box>
  )
}

function formatABC(s: StructureRow): string {
  const lp = (s as unknown as { lattice_parameters?: { a: number; b: number; c: number } })
    .lattice_parameters
  if (!lp) return '—'
  return `${lp.a.toFixed(3)} / ${lp.b.toFixed(3)} / ${lp.c.toFixed(3)} Å`
}

function LatticeSitesTab({ structure }: { structure: StructureRow }) {
  const sites = (structure as unknown as { atoms?: { species: string; position: number[] }[] })
    .atoms
  const lattice = (structure as unknown as { lattice?: { vectors?: number[][] } }).lattice
    ?.vectors
  return (
    <Stack spacing={3}>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Lattice vectors (Å)
          </Typography>
          {lattice ? (
            <Table size="small">
              <TableBody>
                {lattice.map((v, i) => (
                  <TableRow key={i}>
                    <TableCell>{['a', 'b', 'c'][i]}</TableCell>
                    {v.map((x, j) => (
                      <TableCell key={j} align="right">
                        {x.toFixed(4)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <Typography color="text.secondary" variant="body2">
              No lattice data on this record.
            </Typography>
          )}
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Sites
          </Typography>
          {sites && sites.length ? (
            <Table size="small" data-testid="sites-table">
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Element</TableCell>
                  <TableCell align="right">x</TableCell>
                  <TableCell align="right">y</TableCell>
                  <TableCell align="right">z</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sites.map((s, i) => (
                  <TableRow key={i}>
                    <TableCell>{i + 1}</TableCell>
                    <TableCell>{s.species}</TableCell>
                    {s.position.map((x, j) => (
                      <TableCell key={j} align="right">
                        {x.toFixed(4)}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <Typography color="text.secondary" variant="body2">
              No sites on this record.
            </Typography>
          )}
        </CardContent>
      </Card>
    </Stack>
  )
}

function buildViewerInput(s?: StructureRow) {
  if (!s) return null
  const lattice = (s as unknown as { lattice?: { vectors?: number[][] } }).lattice?.vectors
  const atoms = (s as unknown as { atoms?: { species: string; position: number[] }[] }).atoms
  if (!lattice || !atoms || atoms.length === 0) return null
  if (lattice.length !== 3 || lattice.some((v) => v.length !== 3)) return null
  return {
    lattice: lattice as [
      [number, number, number],
      [number, number, number],
      [number, number, number],
    ],
    atoms: atoms.map((a) => ({
      species: a.species,
      position: a.position as [number, number, number],
    })),
    positionsAreCartesian: false,
  }
}
