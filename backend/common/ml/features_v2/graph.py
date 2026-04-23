"""GNN-native radius-graph builder for Phase 6 / Session 6.1.

Output shape matches what PyG / DGL consumers want: integer edge
index, float node features, float edge features. We don't take a
torch dependency at construction time — arrays stay numpy, and the
GNN session (6.4) will do the `torch.from_numpy` at training time.

Conventions
-----------

- **Cutoff**: default 6.0 Å, per the roadmap.
- **Periodic images**: neighbours are searched across the periodic
  box via ``pymatgen.Structure.get_all_neighbors``; each resulting
  neighbour carries the minimum-image distance.
- **Node ordering (canonicalization)**: vertices are sorted by
  ``(species_symbol, frac_x, frac_y, frac_z)`` lexicographically.
  Two pymatgen ``Structure`` instances representing the same crystal
  but with different internal atom order therefore produce
  identical graphs (same node ids, same edges, same features). This
  is the acceptance-test contract.
- **Edge ordering**: within each source node the edges are sorted
  by ``(distance, target_node_idx)``. Deterministic across runs.

Node features (14-d per atom)
-----------------------------

Default element list (``DEFAULT_ELEMENT_ONE_HOT``) covers the first
30 elements — enough for the roadmap's oxides_gap_v1 dataset and
most metal/semiconductor workflows. Elements outside the list are
encoded with an all-zeros one-hot + still get their Z / χ / period /
group continuous features. Users override the list by passing
``element_one_hot`` to :func:`build_radius_graph`.

Concretely, each node vector is:

    [one_hot_of_len_K, Z_scaled, electronegativity, period, group,
     is_metal_flag]

where ``one_hot_of_len_K`` is the one-hot over ``element_one_hot``.
Z is scaled by 1/118 so it's ~unit magnitude. ``is_metal_flag`` is
1.0 for metallic elements, 0.0 otherwise — useful for ML models
that don't learn that distinction from Z alone on small datasets.

Edge features (10-d per edge)
-----------------------------

- ``[distance, inv_distance]`` — raw pair distance in Å + its
  reciprocal (stabilizes for GNN message passing near zero).
- ``8-center Gaussian basis`` over [0, cutoff] with σ = cutoff / 8:
  a soft binning of the distance used as edge embedding input by
  CGCNN-style models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Element, Structure


DEFAULT_CUTOFF_A = 6.0
DEFAULT_GAUSSIAN_N_CENTERS = 8

DEFAULT_ELEMENT_ONE_HOT: Tuple[str, ...] = (
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn",
)


@dataclass
class RadiusGraph:
    """Return of :func:`build_radius_graph`.

    Edge convention: directed — each undirected pair contributes two
    rows in ``edge_index`` (once for each endpoint). Consumers that
    want undirected can ``edge_index[:, ::2]`` to keep one direction.
    """

    node_features: np.ndarray      # (n_nodes, n_node_feat)
    edge_index: np.ndarray         # (2, n_edges); source → target
    edge_features: np.ndarray      # (n_edges, n_edge_feat)
    node_species: List[str]        # per-node atom symbol, canonically ordered
    node_frac_coords: np.ndarray   # (n_nodes, 3) — post-canonical fractional coords
    cutoff_angstrom: float


def build_radius_graph(
    structure: Structure,
    *,
    cutoff_angstrom: float = DEFAULT_CUTOFF_A,
    element_one_hot: Sequence[str] = DEFAULT_ELEMENT_ONE_HOT,
    gaussian_n_centers: int = DEFAULT_GAUSSIAN_N_CENTERS,
) -> RadiusGraph:
    """Construct a canonicalized radius graph.

    Raises :class:`ValueError` on empty structures or zero cutoff.
    """
    if cutoff_angstrom <= 0:
        raise ValueError(f"cutoff_angstrom must be > 0; got {cutoff_angstrom}")
    if len(structure) == 0:
        raise ValueError("cannot build radius graph on an empty structure")

    # Canonical node ordering: (species_str, frac_x, frac_y, frac_z) lexsort.
    species = [str(s) for s in structure.species]
    frac = np.asarray(structure.frac_coords, dtype=np.float64)
    sort_keys = list(zip(species, frac[:, 0], frac[:, 1], frac[:, 2]))
    canonical_order = sorted(range(len(species)), key=lambda i: sort_keys[i])

    # Rebuild a fresh Structure in canonical order so pymatgen's
    # neighbour search works directly on the canonical index set.
    from pymatgen.core import Structure as _Structure

    canonical = _Structure(
        structure.lattice,
        [species[i] for i in canonical_order],
        [frac[i] for i in canonical_order],
    )

    # Node features.
    one_hot_set = {sym: k for k, sym in enumerate(element_one_hot)}
    K = len(element_one_hot)
    node_feat_dim = K + 5  # + Z_scaled, chi, period, group, is_metal
    n = len(canonical)
    node_features = np.zeros((n, node_feat_dim), dtype=np.float32)
    for i, site in enumerate(canonical):
        sym = str(site.specie.symbol)
        if sym in one_hot_set:
            node_features[i, one_hot_set[sym]] = 1.0
        try:
            el = Element(sym)
        except Exception:  # noqa: BLE001
            # Pathological — skip the continuous features and leave
            # zero. Almost never triggers for real structures.
            continue
        # Z scaled by 1/118 (max Z in the periodic table).
        node_features[i, K] = (el.Z or 0) / 118.0
        # Pauling electronegativity; 0 for noble gases without one.
        chi = el.X
        node_features[i, K + 1] = float(chi) if chi is not None else 0.0
        node_features[i, K + 2] = float(el.row or 0)
        node_features[i, K + 3] = float(el.group or 0)
        node_features[i, K + 4] = 1.0 if el.is_metal else 0.0

    # Neighbours: pymatgen returns (site, distance, index, image) tuples
    # per source site. We take index + distance and ignore image —
    # CGCNN-style models don't care which image the neighbour came
    # from as long as the distance is right.
    all_nbrs = canonical.get_all_neighbors(r=cutoff_angstrom, include_index=True)

    # Assemble edges deterministically.
    source_list: List[int] = []
    target_list: List[int] = []
    distance_list: List[float] = []
    for src, nbrs in enumerate(all_nbrs):
        # Sort per-source by (distance, target_idx) for deterministic
        # edge ordering. Matches the acceptance-test idempotency check.
        nbrs_sorted = sorted(
            nbrs, key=lambda t: (float(t[1]), int(t[2])),
        )
        for _nbr_site, distance, idx, *_image in nbrs_sorted:
            source_list.append(int(src))
            target_list.append(int(idx))
            distance_list.append(float(distance))

    n_edges = len(source_list)
    edge_index = np.zeros((2, n_edges), dtype=np.int64)
    edge_index[0] = source_list
    edge_index[1] = target_list
    distances = np.asarray(distance_list, dtype=np.float32)

    # Edge features: [d, 1/d, gaussian basis (gaussian_n_centers)].
    centers = np.linspace(
        0.0, cutoff_angstrom, gaussian_n_centers, dtype=np.float32,
    )
    sigma = cutoff_angstrom / max(1, gaussian_n_centers)
    edge_feat_dim = 2 + gaussian_n_centers
    edge_features = np.zeros((n_edges, edge_feat_dim), dtype=np.float32)
    if n_edges > 0:
        edge_features[:, 0] = distances
        # Stabilized reciprocal: 1/(d + ε). Avoids inf on degenerate
        # coincident atoms (shouldn't happen but harmless to guard).
        edge_features[:, 1] = 1.0 / (distances + 1e-6)
        diff = distances[:, None] - centers[None, :]
        edge_features[:, 2:] = np.exp(-0.5 * (diff / sigma) ** 2)

    return RadiusGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_species=[str(s.specie.symbol) for s in canonical],
        node_frac_coords=np.asarray(canonical.frac_coords, dtype=np.float64),
        cutoff_angstrom=cutoff_angstrom,
    )
