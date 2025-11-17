/**
 * TypeScript types for design and optimization API
 *
 * These types match the backend Pydantic schemas for:
 * - Material design search
 * - Property-based filtering
 * - Candidate structure ranking
 */

/**
 * Constraint for a single material property
 */
export interface PropertyConstraint {
  min?: number;
  max?: number;
  target?: number;
}

/**
 * Request parameters for design-based material search
 */
export interface DesignSearchRequest {
  // Property constraints
  target_bandgap?: PropertyConstraint;
  target_formation_energy?: PropertyConstraint;
  target_stability_score?: PropertyConstraint;

  // Structural constraints
  dimensionality?: 0 | 1 | 2 | 3;
  elements?: string[];
  exclude_elements?: string[];
  max_atoms?: number;
  min_atoms?: number;

  // Search parameters
  limit?: number;
  include_generated?: boolean;
  min_score?: number;
}

/**
 * A candidate structure matching search criteria
 */
export interface CandidateStructure {
  structure_id: string;
  material_id: string;
  formula: string;
  score: number;
  properties: Record<string, number>;
  property_source: 'ML' | 'SIMULATION' | 'MIXED' | 'GENERATED';
  match_details: Record<string, any>;
  dimensionality?: number;
  num_atoms?: number;
  elements?: string[];
  is_generated: boolean;
  parent_structure_id?: string;
  generation_method?: string;
}

/**
 * Response from design search endpoint
 */
export interface DesignSearchResponse {
  candidates: CandidateStructure[];
  total_found: number;
  search_params: Record<string, any>;
  search_time_ms: number;
  score_distribution?: {
    min: number;
    max: number;
    mean: number;
    median: number;
  };
  property_ranges?: Record<
    string,
    {
      min: number;
      max: number;
      mean: number;
      count: number;
    }
  >;
}

/**
 * Statistics for design search capabilities
 */
export interface DesignStats {
  total_structures: number;
  structures_with_predictions: number;
  dimensionality_counts: Record<number, number>;
  coverage: {
    prediction_coverage: number;
  };
}
