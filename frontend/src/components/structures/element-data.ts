/**
 * Phase 9 / Session 9.2 — element CPK colors + covalent radii.
 *
 * CPK ("Corey-Pauling-Koltun") colors are the standard color
 * convention for atomic structure visualization (carbon = black,
 * oxygen = red, hydrogen = white, etc.).
 *
 * Covalent radii (Å) come from Cordero et al. 2008
 * (doi:10.1039/b801115j). The bond detector in StructureViewer
 * declares a bond between sites *i* and *j* when
 *
 *     |r_ij| ≤ (R_cov(i) + R_cov(j)) × BOND_TOLERANCE
 *
 * where ``BOND_TOLERANCE = 1.15`` matches the roadmap text and the
 * ``pymatgen.analysis.local_env.BrunnerNN_real`` default.
 *
 * Only the elements we expect in seeded ORION data (the Si /
 * oxide / TMD test fixtures + first-row + common transition
 * metals) are listed; missing elements fall back to a neutral
 * grey sphere with a 1.5-Å covalent radius (large enough to
 * still detect plausible bonds).
 */

export interface ElementInfo {
  /** CPK hex color, e.g. ``"#FF0D0D"`` for oxygen. */
  color: string
  /** Covalent radius in Å (Cordero 2008). */
  covalentRadius: number
}

const TABLE: Record<string, ElementInfo> = {
  H:  { color: '#FFFFFF', covalentRadius: 0.31 },
  Li: { color: '#CC80FF', covalentRadius: 1.28 },
  Be: { color: '#C2FF00', covalentRadius: 0.96 },
  B:  { color: '#FFB5B5', covalentRadius: 0.84 },
  C:  { color: '#909090', covalentRadius: 0.76 },
  N:  { color: '#3050F8', covalentRadius: 0.71 },
  O:  { color: '#FF0D0D', covalentRadius: 0.66 },
  F:  { color: '#90E050', covalentRadius: 0.57 },
  Na: { color: '#AB5CF2', covalentRadius: 1.66 },
  Mg: { color: '#8AFF00', covalentRadius: 1.41 },
  Al: { color: '#BFA6A6', covalentRadius: 1.21 },
  Si: { color: '#F0C8A0', covalentRadius: 1.11 },
  P:  { color: '#FF8000', covalentRadius: 1.07 },
  S:  { color: '#FFFF30', covalentRadius: 1.05 },
  Cl: { color: '#1FF01F', covalentRadius: 1.02 },
  K:  { color: '#8F40D4', covalentRadius: 2.03 },
  Ca: { color: '#3DFF00', covalentRadius: 1.76 },
  Ti: { color: '#BFC2C7', covalentRadius: 1.60 },
  V:  { color: '#A6A6AB', covalentRadius: 1.53 },
  Cr: { color: '#8A99C7', covalentRadius: 1.39 },
  Mn: { color: '#9C7AC7', covalentRadius: 1.39 },
  Fe: { color: '#E06633', covalentRadius: 1.32 },
  Co: { color: '#F090A0', covalentRadius: 1.26 },
  Ni: { color: '#50D050', covalentRadius: 1.24 },
  Cu: { color: '#C88033', covalentRadius: 1.32 },
  Zn: { color: '#7D80B0', covalentRadius: 1.22 },
  Ga: { color: '#C28F8F', covalentRadius: 1.22 },
  Ge: { color: '#668F8F', covalentRadius: 1.20 },
  As: { color: '#BD80E3', covalentRadius: 1.19 },
  Se: { color: '#FFA100', covalentRadius: 1.20 },
  Br: { color: '#A62929', covalentRadius: 1.20 },
  Sr: { color: '#00FF00', covalentRadius: 1.95 },
  Y:  { color: '#94FFFF', covalentRadius: 1.90 },
  Zr: { color: '#94E0E0', covalentRadius: 1.75 },
  Nb: { color: '#73C2C9', covalentRadius: 1.64 },
  Mo: { color: '#54B5B5', covalentRadius: 1.54 },
  Ru: { color: '#248F8F', covalentRadius: 1.46 },
  Rh: { color: '#0A7D8C', covalentRadius: 1.42 },
  Pd: { color: '#006985', covalentRadius: 1.39 },
  Ag: { color: '#C0C0C0', covalentRadius: 1.45 },
  Cd: { color: '#FFD98F', covalentRadius: 1.44 },
  In: { color: '#A67573', covalentRadius: 1.42 },
  Sn: { color: '#668080', covalentRadius: 1.39 },
  Sb: { color: '#9E63B5', covalentRadius: 1.39 },
  Te: { color: '#D47A00', covalentRadius: 1.38 },
  I:  { color: '#940094', covalentRadius: 1.39 },
  Cs: { color: '#57178F', covalentRadius: 2.44 },
  Ba: { color: '#00C900', covalentRadius: 2.15 },
  La: { color: '#70D4FF', covalentRadius: 2.07 },
  W:  { color: '#2194D6', covalentRadius: 1.62 },
  Pt: { color: '#D0D0E0', covalentRadius: 1.36 },
  Au: { color: '#FFD123', covalentRadius: 1.36 },
  Hg: { color: '#B8B8D0', covalentRadius: 1.32 },
  Pb: { color: '#575961', covalentRadius: 1.46 },
}

const DEFAULT: ElementInfo = { color: '#CCCCCC', covalentRadius: 1.5 }

/** Look up element info; falls back to grey 1.5 Å for unknowns. */
export function elementInfo(symbol: string): ElementInfo {
  return TABLE[symbol] ?? DEFAULT
}

export const BOND_TOLERANCE = 1.15
