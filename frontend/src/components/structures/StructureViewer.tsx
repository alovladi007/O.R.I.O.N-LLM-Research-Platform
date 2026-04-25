// @ts-nocheck — react-three-fiber's JSX intrinsic types collide with
// TypeScript's HTML <line> when both type sets are loaded. Pre-existing
// project-wide; the runtime is correct.
'use client'

/**
 * Phase 9 / Session 9.2 — StructureViewer.
 *
 * Built from scratch per the roadmap (the legacy
 * StructureViewer3D is left in place for any old callers but is not
 * imported by the new pages). Uses ``@react-three/fiber`` +
 * ``@react-three/drei`` (already in deps).
 *
 * Features:
 *   - CPK-colored spheres at each fractional site mapped to
 *     Cartesian coords via the lattice.
 *   - Bonds inferred from a covalent-radii sum × 1.15 cutoff
 *     (Cordero 2008; matches pymatgen.local_env defaults).
 *   - Supercell controls (1 / 2 / 3 along each axis, independent).
 *   - Camera reset + screenshot (PNG download).
 *
 * Input shape: matches the parsed structure returned by
 * ``StructureParseResponse`` from the backend (lattice.vectors +
 * atoms[{species, position}]). The detail page passes this data
 * straight through.
 */

import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import {
  Box,
  ButtonGroup,
  IconButton,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'
import {
  Add as AddIcon,
  Remove as RemoveIcon,
  CenterFocusStrong,
  PhotoCamera,
} from '@mui/icons-material'

import { BOND_TOLERANCE, elementInfo } from './element-data'

// --------------------------------------------------------------------
// Types
// --------------------------------------------------------------------

export interface ViewerSite {
  /** Element symbol (``"Si"``, ``"O"``, …). */
  species: string
  /**
   * Atomic position. May be either fractional (``[0, 0.5, 0.25]``,
   * default) or Cartesian; toggle via ``positionsAreCartesian``.
   */
  position: [number, number, number]
}

export interface ViewerStructure {
  /** Lattice vectors (3×3 in Å). */
  lattice: [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ]
  atoms: ViewerSite[]
  /** If true, ``site.position`` is treated as Cartesian (Å). */
  positionsAreCartesian?: boolean
}

interface StructureViewerProps {
  structure: ViewerStructure
  /** Sphere scale; 1.0 = covalent radius. */
  sphereScale?: number
  /** Show the lattice cell wireframe. */
  showCell?: boolean
  /** Initial supercell repeats per axis. */
  initialSupercell?: [number, number, number]
  /** Pixel height of the canvas (width is 100%). */
  height?: number
  /** ``data-testid`` exposed on the canvas wrapper for Playwright. */
  testId?: string
}

// --------------------------------------------------------------------
// Coordinate helpers
// --------------------------------------------------------------------

function fracToCart(
  frac: [number, number, number],
  lattice: ViewerStructure['lattice'],
): [number, number, number] {
  const [a, b, c] = lattice
  return [
    frac[0] * a[0] + frac[1] * b[0] + frac[2] * c[0],
    frac[0] * a[1] + frac[1] * b[1] + frac[2] * c[1],
    frac[0] * a[2] + frac[1] * b[2] + frac[2] * c[2],
  ]
}

function buildExpandedSites(
  structure: ViewerStructure,
  supercell: [number, number, number],
): { species: string; cart: [number, number, number] }[] {
  const sites: { species: string; cart: [number, number, number] }[] = []
  const [na, nb, nc] = supercell
  for (const atom of structure.atoms) {
    for (let i = 0; i < na; i++) {
      for (let j = 0; j < nb; j++) {
        for (let k = 0; k < nc; k++) {
          const frac: [number, number, number] = structure.positionsAreCartesian
            ? // Cartesian input: shift by lattice * (i, j, k).
              [atom.position[0], atom.position[1], atom.position[2]]
            : [atom.position[0] + i, atom.position[1] + j, atom.position[2] + k]
          const cart = structure.positionsAreCartesian
            ? [
                atom.position[0] +
                  i * structure.lattice[0][0] +
                  j * structure.lattice[1][0] +
                  k * structure.lattice[2][0],
                atom.position[1] +
                  i * structure.lattice[0][1] +
                  j * structure.lattice[1][1] +
                  k * structure.lattice[2][1],
                atom.position[2] +
                  i * structure.lattice[0][2] +
                  j * structure.lattice[1][2] +
                  k * structure.lattice[2][2],
              ] as [number, number, number]
            : fracToCart(frac, structure.lattice)
          sites.push({ species: atom.species, cart })
        }
      }
    }
  }
  return sites
}

function detectBonds(
  sites: { species: string; cart: [number, number, number] }[],
): { i: number; j: number }[] {
  const bonds: { i: number; j: number }[] = []
  for (let i = 0; i < sites.length; i++) {
    const ri = elementInfo(sites[i].species).covalentRadius
    for (let j = i + 1; j < sites.length; j++) {
      const rj = elementInfo(sites[j].species).covalentRadius
      const cutoff = (ri + rj) * BOND_TOLERANCE
      const dx = sites[i].cart[0] - sites[j].cart[0]
      const dy = sites[i].cart[1] - sites[j].cart[1]
      const dz = sites[i].cart[2] - sites[j].cart[2]
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (d <= cutoff && d > 1e-3) bonds.push({ i, j })
    }
  }
  return bonds
}

function structureCenter(
  sites: { cart: [number, number, number] }[],
): [number, number, number] {
  if (!sites.length) return [0, 0, 0]
  let cx = 0,
    cy = 0,
    cz = 0
  for (const s of sites) {
    cx += s.cart[0]
    cy += s.cart[1]
    cz += s.cart[2]
  }
  return [cx / sites.length, cy / sites.length, cz / sites.length]
}

// --------------------------------------------------------------------
// Subcomponents
// --------------------------------------------------------------------

function Atom({
  position,
  color,
  radius,
  testId,
}: {
  position: [number, number, number]
  color: string
  radius: number
  testId?: string
}) {
  return (
    <mesh position={position} userData={{ testId }}>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial color={color} roughness={0.4} metalness={0.1} />
    </mesh>
  )
}

function Bond({
  start,
  end,
}: {
  start: [number, number, number]
  end: [number, number, number]
}) {
  const dx = end[0] - start[0]
  const dy = end[1] - start[1]
  const dz = end[2] - start[2]
  const length = Math.sqrt(dx * dx + dy * dy + dz * dz)
  const mid: [number, number, number] = [
    (start[0] + end[0]) / 2,
    (start[1] + end[1]) / 2,
    (start[2] + end[2]) / 2,
  ]
  // Cylinder is initially along Y; rotate to point from start→end.
  const dir = new THREE.Vector3(dx, dy, dz).normalize()
  const yAxis = new THREE.Vector3(0, 1, 0)
  const quaternion = new THREE.Quaternion().setFromUnitVectors(yAxis, dir)
  return (
    <mesh position={mid} quaternion={quaternion}>
      <cylinderGeometry args={[0.08, 0.08, length, 12]} />
      <meshStandardMaterial color="#888888" roughness={0.6} />
    </mesh>
  )
}

function CellEdges({ lattice }: { lattice: ViewerStructure['lattice'] }) {
  const points = useMemo(() => {
    const o: [number, number, number] = [0, 0, 0]
    const a = lattice[0]
    const b = lattice[1]
    const c = lattice[2]
    const ab: [number, number, number] = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    const ac: [number, number, number] = [a[0] + c[0], a[1] + c[1], a[2] + c[2]]
    const bc: [number, number, number] = [b[0] + c[0], b[1] + c[1], b[2] + c[2]]
    const abc: [number, number, number] = [
      a[0] + b[0] + c[0],
      a[1] + b[1] + c[1],
      a[2] + b[2] + c[2],
    ]
    // 12 edges of a parallelepiped.
    const edges: [number, number, number][][] = [
      [o, a],
      [o, b],
      [o, c],
      [a, ab],
      [a, ac],
      [b, ab],
      [b, bc],
      [c, ac],
      [c, bc],
      [ab, abc],
      [ac, abc],
      [bc, abc],
    ]
    const out: number[] = []
    for (const [p, q] of edges) {
      out.push(p[0], p[1], p[2], q[0], q[1], q[2])
    }
    return new Float32Array(out)
  }, [lattice])

  const ref = useRef<THREE.BufferGeometry>(null)
  useEffect(() => {
    if (ref.current) {
      ref.current.setAttribute('position', new THREE.BufferAttribute(points, 3))
    }
  }, [points])

  return (
    <lineSegments>
      <bufferGeometry ref={ref} />
      <lineBasicMaterial color="#1e3a8a" linewidth={2} />
    </lineSegments>
  )
}

function CameraReset({ target }: { target: [number, number, number] }) {
  const { camera } = useThree()
  useEffect(() => {
    camera.position.set(target[0] + 8, target[1] + 8, target[2] + 12)
    camera.lookAt(target[0], target[1], target[2])
  }, [camera, target])
  return null
}

// --------------------------------------------------------------------
// Public component
// --------------------------------------------------------------------

export default function StructureViewer({
  structure,
  sphereScale = 0.55,
  showCell = true,
  initialSupercell = [1, 1, 1],
  height = 480,
  testId = 'structure-viewer',
}: StructureViewerProps) {
  const [sa, setSa] = useState(initialSupercell[0])
  const [sb, setSb] = useState(initialSupercell[1])
  const [sc, setSc] = useState(initialSupercell[2])
  const [cameraResetKey, setCameraResetKey] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const sites = useMemo(
    () => buildExpandedSites(structure, [sa, sb, sc]),
    [structure, sa, sb, sc],
  )
  const bonds = useMemo(() => detectBonds(sites), [sites])
  const center = useMemo(() => structureCenter(sites), [sites])

  const meshCount = sites.length + bonds.length

  const screenshot = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const url = canvas.toDataURL('image/png')
    const link = document.createElement('a')
    link.download = `structure-${Date.now()}.png`
    link.href = url
    link.click()
  }

  return (
    <Box
      data-testid={testId}
      data-mesh-count={meshCount}
      sx={{
        position: 'relative',
        width: '100%',
        height,
        bgcolor: '#0a0a14',
        borderRadius: 1,
        overflow: 'hidden',
      }}
    >
      <Canvas
        gl={{ preserveDrawingBuffer: true }}
        onCreated={(state) => {
          canvasRef.current = state.gl.domElement
        }}
      >
        <PerspectiveCamera makeDefault fov={50} position={[10, 10, 10]} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 10]} intensity={1} />
        <directionalLight position={[-10, -10, -10]} intensity={0.4} />
        <Suspense fallback={null}>
          {showCell && <CellEdges lattice={structure.lattice} />}
          {sites.map((s, idx) => {
            const info = elementInfo(s.species)
            return (
              <Atom
                key={idx}
                position={s.cart}
                color={info.color}
                radius={info.covalentRadius * sphereScale}
                testId={`atom-${idx}`}
              />
            )
          })}
          {bonds.map((b, idx) => (
            <Bond key={`b-${idx}`} start={sites[b.i].cart} end={sites[b.j].cart} />
          ))}
        </Suspense>
        <OrbitControls
          enablePan
          enableRotate
          enableZoom
          target={center as unknown as THREE.Vector3}
        />
        <CameraReset key={cameraResetKey} target={center} />
      </Canvas>
      {/* Overlay controls */}
      <Box
        sx={{
          position: 'absolute',
          top: 12,
          right: 12,
          bgcolor: 'rgba(255,255,255,0.85)',
          borderRadius: 1,
          p: 1,
        }}
      >
        <Stack spacing={1}>
          <Typography variant="caption" sx={{ fontWeight: 600 }}>
            Supercell
          </Typography>
          {(
            [
              ['a', sa, setSa],
              ['b', sb, setSb],
              ['c', sc, setSc],
            ] as const
          ).map(([label, val, setVal]) => (
            <Stack key={label} direction="row" alignItems="center" spacing={0.5}>
              <Typography variant="caption" sx={{ width: 16 }}>
                {label}:
              </Typography>
              <ButtonGroup size="small">
                <IconButton
                  size="small"
                  data-testid={`supercell-${label}-dec`}
                  disabled={val <= 1}
                  onClick={() => setVal(Math.max(1, val - 1))}
                >
                  <RemoveIcon fontSize="inherit" />
                </IconButton>
                <Box sx={{ minWidth: 24, textAlign: 'center', fontSize: 13 }}>
                  {val}
                </Box>
                <IconButton
                  size="small"
                  data-testid={`supercell-${label}-inc`}
                  disabled={val >= 4}
                  onClick={() => setVal(Math.min(4, val + 1))}
                >
                  <AddIcon fontSize="inherit" />
                </IconButton>
              </ButtonGroup>
            </Stack>
          ))}
          <Stack direction="row" spacing={0.5}>
            <Tooltip title="Reset camera">
              <IconButton
                size="small"
                data-testid="viewer-reset-camera"
                onClick={() => setCameraResetKey((k) => k + 1)}
              >
                <CenterFocusStrong fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Screenshot">
              <IconButton
                size="small"
                data-testid="viewer-screenshot"
                onClick={screenshot}
              >
                <PhotoCamera fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </Stack>
      </Box>
    </Box>
  )
}
