/**
 * TypeScript types for crystal structures and materials
 * Matches backend Pydantic schemas
 */

export interface LatticeVectors {
  a: number[];
  b: number[];
  c: number[];
}

export interface PeriodicSite {
  species: string;
  coords: number[]; // Fractional coordinates [x, y, z]
  properties?: Record<string, any>;
}

export interface Structure {
  id: string;
  formula: string;
  material_name?: string;
  lattice_vectors?: number[][]; // 3x3 matrix
  atomic_species?: string[];
  atomic_positions?: number[][]; // Nx3 fractional coordinates
  a?: number;
  b?: number;
  c?: number;
  alpha?: number;
  beta?: number;
  gamma?: number;
  num_atoms?: number;
  dimensionality?: number;
  space_group?: string;
  space_group_number?: number;
  crystal_system?: string;
  point_group?: string;
  volume?: number;
  density?: number;
  band_gap?: number;
  formation_energy?: number;
  energy_above_hull?: number;
  is_stable?: boolean;
  is_gap_direct?: boolean;
  magnetic_ordering?: string;
  total_magnetization?: number;
  elasticity?: ElasticityData;
  created_at?: string;
  updated_at?: string;
  metadata?: Record<string, any>;
}

export interface ElasticityData {
  elastic_tensor?: number[][];
  bulk_modulus?: number;
  shear_modulus?: number;
  youngs_modulus?: number;
  poissons_ratio?: number;
}

export interface Material {
  id: string;
  material_id: string;
  formula: string;
  structure_id?: string;
  dimensionality?: number;
  elements: string[];
  nelements: number;
  composition: Record<string, number>;
  is_ordered: boolean;
  is_magnetic: boolean;
  band_gap?: number;
  formation_energy_per_atom?: number;
  energy_above_hull?: number;
  is_stable: boolean;
  theoretical: boolean;
  database_ids?: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface StructureListParams {
  skip?: number;
  limit?: number;
  formula?: string;
  elements?: string[];
  dimensionality?: number;
  min_band_gap?: number;
  max_band_gap?: number;
  is_stable?: boolean;
  space_group?: string;
  crystal_system?: string;
  sort_by?: string;
  order?: 'asc' | 'desc';
}

export interface StructureListResponse {
  items: Structure[];
  total: number;
  skip: number;
  limit: number;
}

export interface StructureExportFormat {
  format: 'cif' | 'poscar' | 'xyz' | 'json' | 'xsf';
  primitive?: boolean;
  conventional?: boolean;
}

export interface AtomInfo {
  element: string;
  position: number[];
  fractional_position: number[];
  index: number;
}

export interface BondInfo {
  atom1: number;
  atom2: number;
  length: number;
  type?: string;
}

export interface CrystalSystem {
  name: string;
  constraints: string[];
}

export const CRYSTAL_SYSTEMS: Record<string, CrystalSystem> = {
  cubic: { name: 'Cubic', constraints: ['a=b=c', 'α=β=γ=90°'] },
  tetragonal: { name: 'Tetragonal', constraints: ['a=b≠c', 'α=β=γ=90°'] },
  orthorhombic: { name: 'Orthorhombic', constraints: ['a≠b≠c', 'α=β=γ=90°'] },
  hexagonal: { name: 'Hexagonal', constraints: ['a=b≠c', 'α=β=90°, γ=120°'] },
  trigonal: { name: 'Trigonal', constraints: ['a=b≠c', 'α=β=90°, γ=120°'] },
  monoclinic: { name: 'Monoclinic', constraints: ['a≠b≠c', 'α=γ=90°≠β'] },
  triclinic: { name: 'Triclinic', constraints: ['a≠b≠c', 'α≠β≠γ'] },
};

export interface CoordinateMode {
  type: 'fractional' | 'cartesian';
  label: string;
}

export const COORDINATE_MODES: CoordinateMode[] = [
  { type: 'fractional', label: 'Fractional' },
  { type: 'cartesian', label: 'Cartesian (Å)' },
];
