"""Lightning DataModule for Session 6.4 CGCNN training.

Reads from a :class:`DatasetSnapshot` + a row → pymatgen-Structure
resolver, featurizes each structure via
:func:`features_v2.build_radius_graph`, and batches the graphs for
the :class:`CGCNN` model.

Why the resolver callable
-------------------------

``PropertyRow`` does not carry the structure itself — just
``structure_id``. In a DB-backed run the resolver would SELECT
from the ``structures`` table; in a JSONL-ingest run (like
oxides_gap_v1 today) it reconstructs a ``pymatgen.Structure`` from
row.extras or from a separate structures JSONL. The DataModule
takes the resolver as a constructor arg so we can swap those two
paths without changing the model code.

Batching
--------

CGCNN graphs have variable ``n_atoms`` per sample, so we can't stack
into a dense tensor. Instead, we concatenate the per-graph arrays
into one big tensor and keep a ``graph_index`` vector telling the
mean-pool which atoms belong to which graph. ``edge_index`` gets
offset by ``sum(n_atoms_seen_so_far)`` so node IDs stay unique
across the batch. This is the standard PyG-style batching done by
hand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from backend.common.ml.datasets_v2 import DatasetSnapshot, PropertyRow

logger = logging.getLogger(__name__)


StructureResolver = Callable[[PropertyRow], object]  # → pymatgen.Structure


# ---------------------------------------------------------------------------
# Per-row featurization
# ---------------------------------------------------------------------------


@dataclass
class GraphSample:
    """One graph ready for collation."""

    node_features: np.ndarray      # (N, F_node)
    edge_index: np.ndarray         # (2, E)
    edge_features: np.ndarray      # (E, F_edge)
    y: float                       # regression target


def featurize_row(
    row: PropertyRow,
    resolver: StructureResolver,
    cutoff_angstrom: float,
) -> GraphSample:
    """Convert one row to a :class:`GraphSample` via ``build_radius_graph``.

    The resolver is expected to return a pymatgen ``Structure``;
    anything callable + iterable-of-sites-like with a ``.lattice``
    + ``.frac_coords`` works too (``build_radius_graph`` already
    accepts the dict-form fallback).
    """
    from backend.common.ml.features_v2 import build_radius_graph

    structure = resolver(row)
    graph = build_radius_graph(structure, cutoff_angstrom=cutoff_angstrom)
    return GraphSample(
        node_features=graph.node_features.astype(np.float32, copy=False),
        edge_index=graph.edge_index.astype(np.int64, copy=False),
        edge_features=graph.edge_features.astype(np.float32, copy=False),
        y=float(row.property_value),
    )


# ---------------------------------------------------------------------------
# Torch Dataset wrapper
# ---------------------------------------------------------------------------


class CGCNNDataset(Dataset):
    """Holds pre-computed :class:`GraphSample`s for one split.

    We featurize eagerly at init time because ``build_radius_graph``
    costs ~0.5-3 ms per structure; for a 2000-row oxides_gap_v1 that's
    under 6 s of one-time cost and avoids the DataLoader worker
    start-up overhead on every epoch.
    """

    def __init__(self, samples: List[GraphSample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Collate — concat graphs into a single mega-graph + graph_index
# ---------------------------------------------------------------------------


def collate_graphs(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    """Stack a list of :class:`GraphSample`s into one batch.

    Produces the tensor layout the :class:`CGCNN` forward pass
    expects: node features concatenated, edges offset so nodes are
    globally unique, and a ``graph_index`` vector telling the
    mean-pool which atoms belong to which graph.
    """
    if not batch:
        return {
            "node_features": torch.zeros((0, 0), dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_features": torch.zeros((0, 0), dtype=torch.float32),
            "graph_index": torch.zeros(0, dtype=torch.long),
            "y": torch.zeros(0, dtype=torch.float32),
        }

    node_offsets: List[int] = []
    offset = 0
    for g in batch:
        node_offsets.append(offset)
        offset += g.node_features.shape[0]

    node_features = np.concatenate([g.node_features for g in batch], axis=0)
    edge_features = np.concatenate([g.edge_features for g in batch], axis=0)
    edge_index_parts = []
    graph_index_parts = []
    for gi, (g, off) in enumerate(zip(batch, node_offsets)):
        edge_index_parts.append(g.edge_index + off)
        graph_index_parts.append(
            np.full(g.node_features.shape[0], gi, dtype=np.int64)
        )
    edge_index = (
        np.concatenate(edge_index_parts, axis=1)
        if edge_index_parts else np.zeros((2, 0), dtype=np.int64)
    )
    graph_index = np.concatenate(graph_index_parts, axis=0)
    y = np.asarray([g.y for g in batch], dtype=np.float32)

    return {
        "node_features": torch.from_numpy(node_features),
        "edge_index": torch.from_numpy(edge_index),
        "edge_features": torch.from_numpy(edge_features),
        "graph_index": torch.from_numpy(graph_index),
        "y": torch.from_numpy(y),
    }


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------


class CGCNNDataModule(pl.LightningDataModule):
    """Wire ``DatasetSnapshot`` → train/val/test graph loaders.

    Args
    ----
    rows
        Full corpus that the snapshot was built from. Same iterable
        the user passes to ``orion_dataset.py create --rows-jsonl``.
    snapshot
        Frozen snapshot with the train/val/test row IDs.
    resolver
        Callable that maps a :class:`PropertyRow` to a pymatgen
        ``Structure`` (or equivalent ``build_radius_graph`` input).
    batch_size, num_workers, cutoff_angstrom
        Training hyperparameters; cutoff defaults to the 6 Å used
        by ``features_v2``'s default.
    """

    def __init__(
        self,
        *,
        rows: Sequence[PropertyRow],
        snapshot: DatasetSnapshot,
        resolver: StructureResolver,
        batch_size: int = 32,
        num_workers: int = 0,
        cutoff_angstrom: float = 6.0,
    ):
        super().__init__()
        self.rows = list(rows)
        self.snapshot = snapshot
        self.resolver = resolver
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cutoff_angstrom = cutoff_angstrom

        self._train_samples: Optional[List[GraphSample]] = None
        self._val_samples: Optional[List[GraphSample]] = None
        self._test_samples: Optional[List[GraphSample]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_samples is not None:
            return  # Already featurized
        by_id = {r.row_id: r for r in self.rows}

        def _materialize(ids):
            out = []
            for rid in ids:
                r = by_id.get(rid)
                if r is None:
                    continue
                out.append(
                    featurize_row(r, self.resolver, self.cutoff_angstrom)
                )
            return out

        logger.info(
            "CGCNNDataModule.setup: featurizing %d train + %d val + %d test",
            len(self.snapshot.train_row_ids),
            len(self.snapshot.val_row_ids),
            len(self.snapshot.test_row_ids),
        )
        self._train_samples = _materialize(self.snapshot.train_row_ids)
        self._val_samples = _materialize(self.snapshot.val_row_ids)
        self._test_samples = _materialize(self.snapshot.test_row_ids)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            CGCNNDataset(self._train_samples or []),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            CGCNNDataset(self._val_samples or []),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            CGCNNDataset(self._test_samples or []),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
        )
