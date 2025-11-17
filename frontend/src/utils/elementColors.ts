/**
 * Element colors and atomic radii for 3D structure visualization
 * Uses CPK (Corey-Pauling-Koltun) coloring scheme
 */

export interface ElementData {
  symbol: string;
  name: string;
  color: string; // Hex color
  radius: number; // Van der Waals radius in Angstroms
  atomicNumber: number;
}

/**
 * CPK color scheme for elements
 * Reference: https://en.wikipedia.org/wiki/CPK_coloring
 */
export const ELEMENT_COLORS: Record<string, string> = {
  // Common non-metals
  H: '#FFFFFF',  // White
  C: '#909090',  // Gray
  N: '#3050F8',  // Blue
  O: '#FF0D0D',  // Red
  F: '#90E050',  // Light green
  P: '#FF8000',  // Orange
  S: '#FFFF30',  // Yellow
  Cl: '#1FF01F', // Green
  Br: '#A62929', // Dark red
  I: '#940094',  // Purple

  // Noble gases
  He: '#D9FFFF', // Cyan
  Ne: '#B3E3F5', // Light blue
  Ar: '#80D1E3', // Cyan
  Kr: '#8DD1E3', // Cyan
  Xe: '#429EB0', // Cyan
  Rn: '#420066', // Purple

  // Alkali metals
  Li: '#CC80FF', // Violet
  Na: '#AB5CF2', // Purple
  K: '#8F40D4',  // Purple
  Rb: '#702EB0', // Purple
  Cs: '#57178F', // Purple
  Fr: '#420066', // Purple

  // Alkaline earth metals
  Be: '#C2FF00', // Yellow-green
  Mg: '#8AFF00', // Green
  Ca: '#3DFF00', // Green
  Sr: '#00FF00', // Green
  Ba: '#00D500', // Green
  Ra: '#007D00', // Dark green

  // Transition metals (first row)
  Sc: '#E6E6E6', // Light gray
  Ti: '#BFC2C7', // Gray
  V: '#A6A6AB',  // Gray
  Cr: '#8A99C7', // Steel blue
  Mn: '#9C7AC7', // Purple
  Fe: '#E06633', // Orange-red
  Co: '#F090A0', // Pink
  Ni: '#50D050', // Green
  Cu: '#C88033', // Brown
  Zn: '#7D80B0', // Blue-gray

  // Transition metals (second row)
  Y: '#94FFFF',  // Cyan
  Zr: '#94E0E0', // Cyan
  Nb: '#73C2C9', // Cyan
  Mo: '#54B5B5', // Cyan
  Tc: '#3B9E9E', // Cyan
  Ru: '#248F8F', // Cyan
  Rh: '#0A7D8C', // Cyan
  Pd: '#006985', // Blue
  Ag: '#C0C0C0', // Silver
  Cd: '#FFD98F', // Light orange

  // Transition metals (third row)
  La: '#70D4FF', // Light blue
  Hf: '#4DC2FF', // Blue
  Ta: '#4DA6FF', // Blue
  W: '#2194D6',  // Blue
  Re: '#267DAB', // Blue
  Os: '#266696', // Blue
  Ir: '#175487', // Blue
  Pt: '#D0D0E0', // Light gray
  Au: '#FFD123', // Gold
  Hg: '#B8B8D0', // Gray

  // Post-transition metals
  Al: '#BFA6A6', // Brown-gray
  Ga: '#C28F8F', // Brown
  In: '#A67573', // Brown
  Sn: '#668080', // Gray
  Tl: '#A6544D', // Brown
  Pb: '#575961', // Dark gray
  Bi: '#9E4FB5', // Purple
  Po: '#AB5C00', // Brown

  // Metalloids
  B: '#FFB5B5',  // Pink
  Si: '#F0C8A0', // Tan
  Ge: '#668F8F', // Gray
  As: '#BD80E3', // Purple
  Sb: '#9E63B5', // Purple
  Te: '#D47A00', // Orange
  At: '#754F45', // Brown

  // Lanthanides
  Ce: '#FFFFC7', // Light yellow
  Pr: '#D9FFC7', // Light green
  Nd: '#C7FFC7', // Light green
  Pm: '#A3FFC7', // Green
  Sm: '#8FFFC7', // Green
  Eu: '#61FFC7', // Green
  Gd: '#45FFC7', // Green
  Tb: '#30FFC7', // Green
  Dy: '#1FFFC7', // Cyan
  Ho: '#00FF9C', // Cyan
  Er: '#00E675', // Green
  Tm: '#00D452', // Green
  Yb: '#00BF38', // Green
  Lu: '#00AB24', // Green

  // Actinides
  Ac: '#70ABFA', // Light blue
  Th: '#00BAFF', // Cyan
  Pa: '#00A1FF', // Blue
  U: '#008FFF',  // Blue
  Np: '#0080FF', // Blue
  Pu: '#006BFF', // Blue
  Am: '#545CF2', // Blue
  Cm: '#785CE3', // Purple
  Bk: '#8A4FE3', // Purple
  Cf: '#A136D4', // Purple
  Es: '#B31FD4', // Purple
  Fm: '#B31FBA', // Purple
};

/**
 * Van der Waals radii for elements (in Angstroms)
 * Used for sphere sizing in 3D visualization
 */
export const ATOMIC_RADII: Record<string, number> = {
  H: 1.20,  C: 1.70,  N: 1.55,  O: 1.52,  F: 1.47,
  P: 1.80,  S: 1.80,  Cl: 1.75, Br: 1.85, I: 1.98,
  He: 1.40, Ne: 1.54, Ar: 1.88, Kr: 2.02, Xe: 2.16,
  Li: 1.82, Na: 2.27, K: 2.75,  Rb: 3.03, Cs: 3.43,
  Be: 1.53, Mg: 1.73, Ca: 2.31, Sr: 2.49, Ba: 2.68,
  Sc: 2.11, Ti: 2.00, V: 1.92,  Cr: 1.89, Mn: 1.87,
  Fe: 1.72, Co: 1.67, Ni: 1.63, Cu: 1.40, Zn: 1.39,
  Y: 2.32,  Zr: 2.23, Nb: 2.18, Mo: 2.17, Tc: 2.16,
  Ru: 2.13, Rh: 2.10, Pd: 2.10, Ag: 1.72, Cd: 1.58,
  La: 2.43, Hf: 2.23, Ta: 2.22, W: 2.18,  Re: 2.16,
  Os: 2.16, Ir: 2.13, Pt: 1.75, Au: 1.66, Hg: 1.55,
  Al: 1.84, Ga: 1.87, In: 1.93, Sn: 2.17, Tl: 1.96,
  Pb: 2.02, Bi: 2.07, Po: 1.97, B: 1.92,  Si: 2.10,
  Ge: 2.11, As: 1.85, Sb: 2.06, Te: 2.06, At: 2.02,
};

/**
 * Get color for an element (CPK scheme)
 */
export function getElementColor(element: string): string {
  const normalized = element.charAt(0).toUpperCase() + element.slice(1).toLowerCase();
  return ELEMENT_COLORS[normalized] || '#FF1493'; // Default: hot pink for unknown elements
}

/**
 * Get atomic radius for an element
 */
export function getAtomicRadius(element: string): number {
  const normalized = element.charAt(0).toUpperCase() + element.slice(1).toLowerCase();
  return ATOMIC_RADII[normalized] || 1.70; // Default: carbon radius
}

/**
 * Get scaled radius for visualization (scaled down for better appearance)
 */
export function getVisualRadius(element: string, scaleFactor: number = 0.35): number {
  return getAtomicRadius(element) * scaleFactor;
}

/**
 * Convert hex color to RGB tuple [0-1 range]
 */
export function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16) / 255,
        parseInt(result[2], 16) / 255,
        parseInt(result[3], 16) / 255,
      ]
    : [1, 0, 1]; // Magenta for invalid
}

/**
 * Get element data (color, radius, etc.)
 */
export function getElementData(element: string): {
  color: string;
  radius: number;
  visualRadius: number;
  rgb: [number, number, number];
} {
  const color = getElementColor(element);
  const radius = getAtomicRadius(element);
  const visualRadius = getVisualRadius(element);
  const rgb = hexToRgb(color);

  return { color, radius, visualRadius, rgb };
}

/**
 * Get unique elements from a structure
 */
export function getUniqueElements(atomicSpecies?: string[]): string[] {
  if (!atomicSpecies) return [];
  return [...new Set(atomicSpecies)].sort();
}

/**
 * Create element legend data
 */
export interface LegendItem {
  element: string;
  color: string;
  count: number;
}

export function createElementLegend(atomicSpecies?: string[]): LegendItem[] {
  if (!atomicSpecies) return [];

  const counts: Record<string, number> = {};
  atomicSpecies.forEach(element => {
    counts[element] = (counts[element] || 0) + 1;
  });

  return Object.entries(counts)
    .map(([element, count]) => ({
      element,
      color: getElementColor(element),
      count,
    }))
    .sort((a, b) => b.count - a.count); // Sort by count descending
}
