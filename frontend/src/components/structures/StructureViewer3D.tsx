'use client';

import React, { useState, useMemo, useRef, Suspense } from 'react';
import { Canvas, useFrame, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Grid, Html, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  RestartAlt,
  GridOn,
  GridOff,
  Fullscreen,
} from '@mui/icons-material';
import { getElementColor, getVisualRadius, createElementLegend } from '@/utils/elementColors';
import { Structure, AtomInfo } from '@/types/structures';

interface StructureViewerProps {
  structure: Structure;
  width?: number;
  height?: number;
}

interface AtomProps {
  position: [number, number, number];
  element: string;
  index: number;
  onClick: (info: AtomInfo) => void;
  isSelected: boolean;
}

/**
 * Individual atom component with interaction
 */
const Atom: React.FC<AtomProps> = ({ position, element, index, onClick, isSelected }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = getElementColor(element);
  const radius = getVisualRadius(element);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    onClick({
      element,
      position,
      fractional_position: position, // Will be updated by parent
      index,
    });
  };

  useFrame(() => {
    if (meshRef.current) {
      const scale = isSelected ? 1.3 : hovered ? 1.15 : 1.0;
      meshRef.current.scale.lerp(new THREE.Vector3(scale, scale, scale), 0.1);
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={handleClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[radius, 32, 32]} />
      <meshStandardMaterial
        color={color}
        metalness={0.3}
        roughness={0.4}
        emissive={isSelected ? color : '#000000'}
        emissiveIntensity={isSelected ? 0.3 : 0}
      />
      {hovered && (
        <Html distanceFactor={10}>
          <Paper
            elevation={3}
            sx={{
              p: 1,
              bgcolor: 'background.paper',
              pointerEvents: 'none',
            }}
          >
            <Typography variant="caption" fontWeight="bold">
              {element}
            </Typography>
          </Paper>
        </Html>
      )}
    </mesh>
  );
};

/**
 * Unit cell wireframe
 */
const UnitCell: React.FC<{ latticeVectors: number[][] }> = ({ latticeVectors }) => {
  const lines = useMemo(() => {
    const [a, b, c] = latticeVectors;
    const origin = [0, 0, 0];

    // 12 edges of parallelepiped
    const edges = [
      // Base (at origin)
      [origin, a],
      [origin, b],
      [origin, c],
      // Opposite edges
      [a, [a[0] + b[0], a[1] + b[1], a[2] + b[2]]],
      [a, [a[0] + c[0], a[1] + c[1], a[2] + c[2]]],
      [b, [b[0] + a[0], b[1] + a[1], b[2] + a[2]]],
      [b, [b[0] + c[0], b[1] + c[1], b[2] + c[2]]],
      [c, [c[0] + a[0], c[1] + a[1], c[2] + a[2]]],
      [c, [c[0] + b[0], c[1] + b[1], c[2] + b[2]]],
      // Far corner edges
      [[a[0] + b[0], a[1] + b[1], a[2] + b[2]], [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]],
      [[a[0] + c[0], a[1] + c[1], a[2] + c[2]], [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]],
      [[b[0] + c[0], b[1] + c[1], b[2] + c[2]], [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]],
    ];

    return edges;
  }, [latticeVectors]);

  return (
    <>
      {lines.map((edge, i) => {
        const points = [
          new THREE.Vector3(...(edge[0] as [number, number, number])),
          new THREE.Vector3(...(edge[1] as [number, number, number])),
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        return (
          <line key={i} geometry={geometry}>
            <lineBasicMaterial color="#00BFFF" linewidth={2} />
          </line>
        );
      })}
    </>
  );
};

/**
 * 3D scene with atoms and unit cell
 */
const Scene: React.FC<{
  structure: Structure;
  coordinateMode: 'fractional' | 'cartesian';
  showGrid: boolean;
  selectedAtomIndex: number | null;
  onAtomClick: (info: AtomInfo) => void;
}> = ({ structure, coordinateMode, showGrid, selectedAtomIndex, onAtomClick }) => {
  const { atomicPositions, latticeVectors, atomicSpecies } = useMemo(() => {
    const positions = structure.atomic_positions || [];
    const vectors = structure.lattice_vectors || [
      [structure.a || 10, 0, 0],
      [0, structure.b || 10, 0],
      [0, 0, structure.c || 10],
    ];
    const species = structure.atomic_species || [];

    return {
      atomicPositions: positions,
      latticeVectors: vectors,
      atomicSpecies: species,
    };
  }, [structure]);

  // Convert fractional to Cartesian coordinates
  const cartesianPositions = useMemo(() => {
    if (!atomicPositions.length || !latticeVectors.length) return [];

    return atomicPositions.map((frac) => {
      const [fx, fy, fz] = frac;
      const [a, b, c] = latticeVectors;

      return [
        fx * a[0] + fy * b[0] + fz * c[0],
        fx * a[1] + fy * b[1] + fz * c[1],
        fx * a[2] + fy * b[2] + fz * c[2],
      ] as [number, number, number];
    });
  }, [atomicPositions, latticeVectors]);

  // Calculate structure center for camera positioning
  const structureCenter = useMemo(() => {
    if (cartesianPositions.length === 0) return [0, 0, 0];

    const sum = cartesianPositions.reduce(
      (acc, pos) => [acc[0] + pos[0], acc[1] + pos[1], acc[2] + pos[2]],
      [0, 0, 0]
    );

    return [
      sum[0] / cartesianPositions.length,
      sum[1] / cartesianPositions.length,
      sum[2] / cartesianPositions.length,
    ] as [number, number, number];
  }, [cartesianPositions]);

  if (!atomicPositions.length) {
    return (
      <Html center>
        <Alert severity="warning">No atomic positions available</Alert>
      </Html>
    );
  }

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.5} />
      <pointLight position={[0, 0, 0]} intensity={0.3} />

      {/* Camera */}
      <PerspectiveCamera
        makeDefault
        position={[
          structureCenter[0] + 20,
          structureCenter[1] + 20,
          structureCenter[2] + 20,
        ]}
        fov={50}
      />

      {/* Grid */}
      {showGrid && (
        <Grid
          args={[50, 50]}
          cellSize={1}
          cellThickness={0.5}
          cellColor="#6e6e6e"
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#9d4b4b"
          fadeDistance={100}
          fadeStrength={1}
          followCamera={false}
        />
      )}

      {/* Unit Cell */}
      {latticeVectors && <UnitCell latticeVectors={latticeVectors} />}

      {/* Atoms */}
      {cartesianPositions.map((pos, i) => (
        <Atom
          key={i}
          position={pos}
          element={atomicSpecies[i] || 'C'}
          index={i}
          onClick={(info) => {
            onAtomClick({
              ...info,
              fractional_position: atomicPositions[i],
              position: pos,
            });
          }}
          isSelected={selectedAtomIndex === i}
        />
      ))}

      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        target={structureCenter}
      />
    </>
  );
};

/**
 * Main StructureViewer3D component
 */
export const StructureViewer3D: React.FC<StructureViewerProps> = ({
  structure,
  width = 800,
  height = 600,
}) => {
  const [coordinateMode, setCoordinateMode] = useState<'fractional' | 'cartesian'>(
    'fractional'
  );
  const [showGrid, setShowGrid] = useState(true);
  const [selectedAtom, setSelectedAtom] = useState<AtomInfo | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  const legend = useMemo(
    () => createElementLegend(structure.atomic_species),
    [structure.atomic_species]
  );

  const handleFullscreen = () => {
    if (canvasRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        canvasRef.current.requestFullscreen();
      }
    }
  };

  const handleReset = () => {
    setSelectedAtom(null);
  };

  return (
    <Box>
      {/* Controls */}
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <ToggleButtonGroup
            value={coordinateMode}
            exclusive
            onChange={(_, value) => value && setCoordinateMode(value)}
            size="small"
          >
            <ToggleButton value="fractional">Fractional</ToggleButton>
            <ToggleButton value="cartesian">Cartesian</ToggleButton>
          </ToggleButtonGroup>

          <Tooltip title={showGrid ? 'Hide Grid' : 'Show Grid'}>
            <IconButton onClick={() => setShowGrid(!showGrid)} size="small">
              {showGrid ? <GridOff /> : <GridOn />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Reset View">
            <IconButton onClick={handleReset} size="small">
              <RestartAlt />
            </IconButton>
          </Tooltip>

          <Tooltip title="Fullscreen">
            <IconButton onClick={handleFullscreen} size="small">
              <Fullscreen />
            </IconButton>
          </Tooltip>

          <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto' }}>
            {structure.num_atoms || structure.atomic_positions?.length || 0} atoms
          </Typography>
        </Stack>

        {/* Selected Atom Info */}
        {selectedAtom && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'action.hover', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Selected Atom
            </Typography>
            <Stack direction="row" spacing={2}>
              <Chip
                label={selectedAtom.element}
                size="small"
                sx={{
                  bgcolor: getElementColor(selectedAtom.element),
                  color: '#fff',
                  fontWeight: 'bold',
                }}
              />
              <Typography variant="body2">
                Fractional:{' '}
                {selectedAtom.fractional_position
                  .map((v) => v.toFixed(4))
                  .join(', ')}
              </Typography>
              <Typography variant="body2">
                Cartesian: {selectedAtom.position.map((v) => v.toFixed(3)).join(', ')} Å
              </Typography>
            </Stack>
          </Box>
        )}
      </Paper>

      {/* 3D Canvas */}
      <Box ref={canvasRef} sx={{ position: 'relative' }}>
        <Paper
          elevation={3}
          sx={{
            width: '100%',
            height: height,
            overflow: 'hidden',
            bgcolor: '#1a1a2e',
          }}
        >
          <Canvas>
            <Suspense
              fallback={
                <Html center>
                  <CircularProgress />
                </Html>
              }
            >
              <Scene
                structure={structure}
                coordinateMode={coordinateMode}
                showGrid={showGrid}
                selectedAtomIndex={selectedAtom?.index || null}
                onAtomClick={setSelectedAtom}
              />
            </Suspense>
          </Canvas>
        </Paper>

        {/* Element Legend */}
        <Paper
          elevation={3}
          sx={{
            position: 'absolute',
            top: 16,
            right: 16,
            p: 2,
            maxWidth: 200,
            bgcolor: 'rgba(255, 255, 255, 0.95)',
          }}
        >
          <Typography variant="subtitle2" gutterBottom fontWeight="bold">
            Elements
          </Typography>
          <Stack spacing={1}>
            {legend.map((item) => (
              <Stack key={item.element} direction="row" spacing={1} alignItems="center">
                <Box
                  sx={{
                    width: 16,
                    height: 16,
                    borderRadius: '50%',
                    bgcolor: item.color,
                    border: '1px solid #ccc',
                  }}
                />
                <Typography variant="body2">
                  {item.element} ({item.count})
                </Typography>
              </Stack>
            ))}
          </Stack>
        </Paper>
      </Box>
    </Box>
  );
};

export default StructureViewer3D;
