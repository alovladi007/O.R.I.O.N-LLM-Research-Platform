'use client';

/**
 * Material Design and Optimization Page
 *
 * This page provides:
 * - Property-based material search
 * - Multi-constraint filtering
 * - Candidate structure ranking
 * - Rule-based variant generation
 */

import React, { useState, useEffect } from 'react';
import { searchDesigns, getDesignStats } from '@/lib/api';
import {
  DesignSearchRequest,
  DesignSearchResponse,
  CandidateStructure,
  PropertyConstraint,
  DesignStats,
} from '@/types/design';

export default function DesignPage() {
  // Search parameters
  const [searchParams, setSearchParams] = useState<DesignSearchRequest>({
    limit: 20,
    include_generated: false,
    min_score: 0.5,
  });

  // Results
  const [results, setResults] = useState<DesignSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Stats
  const [stats, setStats] = useState<DesignStats | null>(null);

  // Load stats on mount
  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await getDesignStats();
      setStats(data);
    } catch (err: any) {
      console.error('Failed to load stats:', err);
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await searchDesigns(searchParams);
      setResults(response);
    } catch (err: any) {
      setError(err.message || 'Search failed');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const updatePropertyConstraint = (
    property: 'target_bandgap' | 'target_formation_energy' | 'target_stability_score',
    field: 'min' | 'max' | 'target',
    value: string
  ) => {
    const numValue = value === '' ? undefined : parseFloat(value);
    setSearchParams((prev) => ({
      ...prev,
      [property]: {
        ...(prev[property] || {}),
        [field]: numValue,
      },
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Material Design & Optimization
          </h1>
          <p className="text-gray-600">
            Search for materials matching specific design criteria and discover new
            candidates
          </p>

          {/* Statistics */}
          {stats && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-sm text-gray-500">Total Structures</div>
                <div className="text-2xl font-bold text-gray-900">
                  {stats.total_structures.toLocaleString()}
                </div>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-sm text-gray-500">With Predictions</div>
                <div className="text-2xl font-bold text-gray-900">
                  {stats.structures_with_predictions.toLocaleString()}
                </div>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <div className="text-sm text-gray-500">Coverage</div>
                <div className="text-2xl font-bold text-gray-900">
                  {(stats.coverage.prediction_coverage * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Search Form */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">
                Search Criteria
              </h2>

              {/* Property Constraints */}
              <div className="space-y-6">
                {/* Bandgap */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Bandgap (eV)
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-xs text-gray-500">Min</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="0.0"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_bandgap',
                            'min',
                            e.target.value
                          )
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Max</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="10.0"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_bandgap',
                            'max',
                            e.target.value
                          )
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Target</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="2.0"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_bandgap',
                            'target',
                            e.target.value
                          )
                        }
                      />
                    </div>
                  </div>
                </div>

                {/* Formation Energy */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Formation Energy (eV/atom)
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-xs text-gray-500">Min</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="-5.0"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_formation_energy',
                            'min',
                            e.target.value
                          )
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Max</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="0.0"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_formation_energy',
                            'max',
                            e.target.value
                          )
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Target</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="-2.5"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_formation_energy',
                            'target',
                            e.target.value
                          )
                        }
                      />
                    </div>
                  </div>
                </div>

                {/* Stability Score */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Stability Score (0-1)
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-gray-500">Min</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        placeholder="0.7"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_stability_score',
                            'min',
                            e.target.value
                          )
                        }
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Target</label>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        max="1"
                        placeholder="0.9"
                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                        onChange={(e) =>
                          updatePropertyConstraint(
                            'target_stability_score',
                            'target',
                            e.target.value
                          )
                        }
                      />
                    </div>
                  </div>
                </div>

                {/* Structural Constraints */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Dimensionality
                  </label>
                  <select
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                    onChange={(e) =>
                      setSearchParams({
                        ...searchParams,
                        dimensionality:
                          e.target.value === ''
                            ? undefined
                            : (parseInt(e.target.value) as 0 | 1 | 2 | 3),
                      })
                    }
                  >
                    <option value="">Any</option>
                    <option value="0">0D (Molecule)</option>
                    <option value="1">1D (Wire/Tube)</option>
                    <option value="2">2D (Sheet/Layer)</option>
                    <option value="3">3D (Bulk)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Elements (comma-separated)
                  </label>
                  <input
                    type="text"
                    placeholder="e.g., Mo, S"
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                    onChange={(e) =>
                      setSearchParams({
                        ...searchParams,
                        elements:
                          e.target.value === ''
                            ? undefined
                            : e.target.value.split(',').map((s) => s.trim()),
                      })
                    }
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Atoms
                  </label>
                  <input
                    type="number"
                    placeholder="50"
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                    onChange={(e) =>
                      setSearchParams({
                        ...searchParams,
                        max_atoms:
                          e.target.value === '' ? undefined : parseInt(e.target.value),
                      })
                    }
                  />
                </div>

                {/* Search Options */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Limit Results
                  </label>
                  <input
                    type="number"
                    value={searchParams.limit}
                    min="1"
                    max="100"
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                    onChange={(e) =>
                      setSearchParams({
                        ...searchParams,
                        limit: parseInt(e.target.value) || 20,
                      })
                    }
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Minimum Score
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="1"
                    value={searchParams.min_score}
                    className="w-full px-3 py-2 border border-gray-300 rounded"
                    onChange={(e) =>
                      setSearchParams({
                        ...searchParams,
                        min_score: parseFloat(e.target.value) || 0,
                      })
                    }
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="include_generated"
                    checked={searchParams.include_generated}
                    onChange={(e) =>
                      setSearchParams({
                        ...searchParams,
                        include_generated: e.target.checked,
                      })
                    }
                    className="mr-2"
                  />
                  <label htmlFor="include_generated" className="text-sm text-gray-700">
                    Include generated variants
                  </label>
                </div>

                {/* Search Button */}
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {loading ? 'Searching...' : 'Search Materials'}
                </button>
              </div>
            </div>
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">
                Search Results
              </h2>

              {/* Error */}
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
                  {error}
                </div>
              )}

              {/* Loading */}
              {loading && (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                  <p className="mt-4 text-gray-600">Searching materials...</p>
                </div>
              )}

              {/* Results Summary */}
              {results && !loading && (
                <>
                  <div className="mb-4 p-4 bg-gray-50 rounded">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-gray-500">Found</div>
                        <div className="font-bold text-lg">{results.total_found}</div>
                      </div>
                      <div>
                        <div className="text-gray-500">Showing</div>
                        <div className="font-bold text-lg">
                          {results.candidates.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-500">Search Time</div>
                        <div className="font-bold text-lg">
                          {results.search_time_ms.toFixed(0)}ms
                        </div>
                      </div>
                      {results.score_distribution && (
                        <div>
                          <div className="text-gray-500">Best Score</div>
                          <div className="font-bold text-lg">
                            {(results.score_distribution.max * 100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Candidates */}
                  {results.candidates.length === 0 ? (
                    <div className="text-center py-12">
                      <p className="text-gray-500 mb-4">
                        No materials found matching your criteria.
                      </p>
                      <p className="text-sm text-gray-400">
                        Try relaxing constraints or including generated variants.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {results.candidates.map((candidate, index) => (
                        <CandidateCard
                          key={`${candidate.structure_id}-${index}`}
                          candidate={candidate}
                          rank={index + 1}
                        />
                      ))}
                    </div>
                  )}
                </>
              )}

              {/* Empty State */}
              {!results && !loading && !error && (
                <div className="text-center py-12">
                  <p className="text-gray-500">
                    Configure search criteria and click "Search Materials" to find
                    candidates.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Candidate Structure Card Component
 */
function CandidateCard({
  candidate,
  rank,
}: {
  candidate: CandidateStructure;
  rank: number;
}) {
  const scoreColor =
    candidate.score >= 0.8
      ? 'bg-green-100 text-green-800'
      : candidate.score >= 0.6
      ? 'bg-yellow-100 text-yellow-800'
      : 'bg-red-100 text-red-800';

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center font-bold">
            {rank}
          </div>
          <div>
            <h3 className="text-lg font-bold text-gray-900">
              {candidate.formula}
              {candidate.is_generated && (
                <span className="ml-2 text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">
                  Generated
                </span>
              )}
            </h3>
            <div className="text-sm text-gray-500">
              ID: {candidate.structure_id.substring(0, 8)}...
            </div>
          </div>
        </div>
        <div className={`px-3 py-1 rounded font-bold ${scoreColor}`}>
          {(candidate.score * 100).toFixed(1)}%
        </div>
      </div>

      {/* Properties */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
        {candidate.properties.bandgap !== undefined && (
          <div>
            <div className="text-xs text-gray-500">Bandgap</div>
            <div className="font-semibold">
              {candidate.properties.bandgap.toFixed(2)} eV
            </div>
          </div>
        )}
        {candidate.properties.formation_energy !== undefined && (
          <div>
            <div className="text-xs text-gray-500">Formation Energy</div>
            <div className="font-semibold">
              {candidate.properties.formation_energy.toFixed(2)} eV/atom
            </div>
          </div>
        )}
        {candidate.properties.stability_score !== undefined && (
          <div>
            <div className="text-xs text-gray-500">Stability</div>
            <div className="font-semibold">
              {(candidate.properties.stability_score * 100).toFixed(1)}%
            </div>
          </div>
        )}
        {candidate.dimensionality !== undefined && (
          <div>
            <div className="text-xs text-gray-500">Dimensionality</div>
            <div className="font-semibold">{candidate.dimensionality}D</div>
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="px-2 py-1 bg-gray-100 rounded">
          {candidate.property_source}
        </span>
        {candidate.num_atoms && <span>{candidate.num_atoms} atoms</span>}
        {candidate.elements && (
          <span>Elements: {candidate.elements.join(', ')}</span>
        )}
      </div>

      {/* Actions */}
      <div className="mt-3 flex gap-2">
        <a
          href={`/structures/${candidate.structure_id}`}
          className="text-sm text-blue-600 hover:underline"
        >
          View Structure
        </a>
        <span className="text-gray-300">|</span>
        <a
          href={`/materials/${candidate.material_id}`}
          className="text-sm text-blue-600 hover:underline"
        >
          View Material
        </a>
      </div>
    </div>
  );
}
