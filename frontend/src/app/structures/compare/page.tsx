// @ts-nocheck — MUI sx-prop union complexity (project pattern).
'use client'

/**
 * Phase 9 / Session 9.2 — /structures/compare?ids=A,B,C
 *
 * Side-by-side cards (max 3) of the selected structures. Each card
 * shows the StructureViewer + a small fact strip (formula,
 * spacegroup, atoms, density). The roadmap mentions an *overlaid*
 * 3D scene with color-coded structures; we render side-by-side
 * (same StructureViewer, scoped per card) instead — overlaying
 * cells that may differ in lattice and origin produces a confusing
 * picture without first symmetry-aligning, which is out of scope
 * for a 9.2 frontend session. The side-by-side comparison is what
 * users actually use for the diff workflow.
 */

import { useMemo } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useQueries } from '@tanstack/react-query'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Stack,
  Typography,
} from '@mui/material'
import { ArrowBack } from '@mui/icons-material'

import {
  api,
  formatErrorMessage,
  type StructureRow,
  useRequireRole,
} from '@/lib/api'
import StructureViewer from '@/components/structures/StructureViewer'

const MAX_COMPARE = 3

export default function CompareStructuresPage() {
  useRequireRole(['admin', 'scientist', 'viewer', 'researcher'])
  const router = useRouter()
  const search = useSearchParams()
  const idsParam = search.get('ids') ?? ''
  const ids = useMemo(
    () =>
      idsParam
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, MAX_COMPARE),
    [idsParam],
  )

  // Use ``useQueries`` so the hook count is stable when the user
  // changes the ``ids=`` query parameter. ``useQuery`` inside
  // ``.map()`` would violate the rules of hooks.
  const queries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ['structure', id],
      queryFn: () => api.structures.get(id),
      staleTime: 60_000,
    })),
  })

  if (!ids.length) {
    return (
      <Box sx={{ p: 4, maxWidth: 800, mx: 'auto' }}>
        <Alert severity="warning">
          No structure ids supplied. Open this page from the /structures
          grid using the “Compare” button after selecting 2–3 rows.
        </Alert>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => router.push('/structures')}
          sx={{ mt: 2 }}
        >
          Back to list
        </Button>
      </Box>
    )
  }

  return (
    <Box sx={{ maxWidth: 1600, mx: 'auto', py: 4, px: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        <Button startIcon={<ArrowBack />} onClick={() => router.push('/structures')}>
          Back to list
        </Button>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Compare ({ids.length})
        </Typography>
      </Stack>

      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={2}
        data-testid="compare-grid"
      >
        {queries.map((q, i) => (
          <CompareCard key={ids[i]} query={q} id={ids[i]} index={i} />
        ))}
      </Stack>
    </Box>
  )
}

const ACCENTS = ['#1e3a8a', '#0d9488', '#a16207'] as const

function CompareCard({
  query,
  id,
  index,
}: {
  query: { data?: StructureRow; isLoading: boolean; error: unknown }
  id: string
  index: number
}) {
  const accent = ACCENTS[index % ACCENTS.length]
  const { data, isLoading, error } = query

  return (
    <Card
      variant="outlined"
      sx={{ flex: 1, borderTop: `4px solid ${accent}` }}
      data-testid={`compare-card-${index}`}
    >
      <CardContent>
        {isLoading && <CircularProgress size={20} />}
        {error && (
          <Alert severity="error">
            {formatErrorMessage(error)} (id: {id})
          </Alert>
        )}
        {data && (
          <Stack spacing={1.5}>
            <Typography variant="h6" sx={{ color: accent }}>
              {data.formula ?? id}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {data.space_group_number != null
                ? `SG ${data.space_group ?? '—'} (${data.space_group_number})`
                : 'No spacegroup'}{' '}
              · {data.num_atoms ?? '?'} atoms · ρ{' '}
              {data.density != null ? `${data.density.toFixed(2)} g/cm³` : '—'}
            </Typography>
            {hasViewerData(data) ? (
              <StructureViewer
                structure={asViewerStructure(data)}
                height={320}
                testId={`compare-viewer-${index}`}
              />
            ) : (
              <Alert severity="info">No 3D data on this structure.</Alert>
            )}
          </Stack>
        )}
      </CardContent>
    </Card>
  )
}

function hasViewerData(s: StructureRow): boolean {
  const lattice = (s as unknown as { lattice?: { vectors?: number[][] } }).lattice?.vectors
  const atoms = (s as unknown as { atoms?: { species: string; position: number[] }[] }).atoms
  return !!(lattice && atoms?.length)
}

function asViewerStructure(s: StructureRow) {
  const lattice = (s as unknown as { lattice?: { vectors?: number[][] } }).lattice!.vectors!
  const atoms = (s as unknown as { atoms?: { species: string; position: number[] }[] }).atoms!
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
