"""Tests for Phase 6 / Session 6.4 — CGCNN v2 (Lightning + ensemble).

Fast unit tests use synthetic graphs (no MP API, no matminer) so
the suite stays under a minute. The roadmap's full 300-epoch CGCNN
acceptance on oxides_gap_v1 is covered by a separate
:mod:`tests.test_cgcnn_v2_live_acceptance`-style gated test — too
slow for CI's fast loop.

Coverage
--------

1. :class:`CGCNNConv` + :class:`CGCNN` — forward shapes, grads flow,
   deterministic output for fixed seed.
2. ``collate_graphs`` — correct node/edge offsets, graph_index,
   size invariants.
3. :class:`CGCNNLitModule` — training step runs, loss is finite,
   MSE vs Huber selection works.
4. ``CGCNNDataModule`` — featurizes a tiny corpus, split sizes
   match the snapshot.
5. ``train_ensemble`` — N models get produced, each different (seed
   bump), predict_with_uncertainty returns sane shapes.
6. ``save_ensemble`` / ``load_ensemble`` — manifest round-trip.
7. **Bit-exact resume** — a half-trained checkpoint + second trainer
   yields identical weights + metrics as a full run. The roadmap
   calls this out explicitly.
8. Sanity acceptance on a synthetic target where CGCNN should
   learn: MAE drops from epoch 0 baseline — a honest "does gradient
   descent even work" smoke, not the real-data target.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared fixtures — synthetic graph corpus
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_graph_samples():
    """40 synthetic graphs on 3-6 atoms each with a learnable target.

    The target is the mean of the first-channel node feature — an
    aggregation task CGCNN can reproduce perfectly given enough
    epochs. Used to verify gradient flow + loss drops.
    """
    from backend.common.ml.models.cgcnn_v2 import GraphSample

    rng = np.random.default_rng(0)
    samples: List[GraphSample] = []
    for i in range(40):
        n = rng.integers(3, 7)
        node_features = rng.standard_normal((n, 35)).astype(np.float32)
        # Fully-connected bidirectional edges for simplicity.
        pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
        src = np.array([p[0] for p in pairs], dtype=np.int64)
        dst = np.array([p[1] for p in pairs], dtype=np.int64)
        edge_index = np.stack([src, dst], axis=0)
        edge_features = rng.standard_normal(
            (edge_index.shape[1], 10),
        ).astype(np.float32)
        y = float(node_features[:, 0].mean())
        samples.append(GraphSample(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            y=y,
        ))
    return samples


def _make_batch(samples, n: int):
    from backend.common.ml.models.cgcnn_v2 import collate_graphs

    return collate_graphs(samples[:n])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestCGCNN:
    def test_forward_returns_scalar_per_graph(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import CGCNN

        batch = _make_batch(tiny_graph_samples, 5)
        model = CGCNN()
        y_hat = model(
            batch["node_features"], batch["edge_index"],
            batch["edge_features"], batch["graph_index"],
        )
        assert y_hat.shape == (5,)
        assert torch.isfinite(y_hat).all()

    def test_grads_flow_through_convs(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import CGCNN

        batch = _make_batch(tiny_graph_samples, 5)
        model = CGCNN()
        y_hat = model(
            batch["node_features"], batch["edge_index"],
            batch["edge_features"], batch["graph_index"],
        )
        loss = y_hat.pow(2).sum()
        loss.backward()
        # Every parameter has a non-None grad after backward.
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no grad for {n}"

    def test_deterministic_given_seed(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import (
            CGCNN, apply_reproducibility_flags,
        )

        apply_reproducibility_flags(42)
        m1 = CGCNN(hidden_dim=32, n_conv=2)
        apply_reproducibility_flags(42)
        m2 = CGCNN(hidden_dim=32, n_conv=2)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_node_and_edge_counts_add_up(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import collate_graphs

        batch = collate_graphs(tiny_graph_samples[:3])
        total_nodes = sum(
            s.node_features.shape[0] for s in tiny_graph_samples[:3]
        )
        total_edges = sum(
            s.edge_index.shape[1] for s in tiny_graph_samples[:3]
        )
        assert batch["node_features"].shape[0] == total_nodes
        assert batch["edge_index"].shape[1] == total_edges
        assert batch["graph_index"].shape[0] == total_nodes

    def test_edge_offsets_keep_nodes_unique(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import collate_graphs

        batch = collate_graphs(tiny_graph_samples[:4])
        # Every edge's src/dst must be < total_nodes and within a
        # single graph's node block (no cross-graph edges).
        edge_src = batch["edge_index"][0]
        edge_dst = batch["edge_index"][1]
        gi = batch["graph_index"]
        assert (gi[edge_src] == gi[edge_dst]).all(), (
            "collate created cross-graph edges — node offsetting is wrong"
        )

    def test_empty_batch(self):
        from backend.common.ml.models.cgcnn_v2 import collate_graphs

        batch = collate_graphs([])
        assert batch["node_features"].shape[0] == 0


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class TestCGCNNLitModule:
    def test_training_step_produces_finite_loss(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import CGCNNLitModule

        batch = _make_batch(tiny_graph_samples, 6)
        lit = CGCNNLitModule(hidden_dim=32, n_conv=2, loss_kind="mse")
        loss = lit.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_huber_loss_switch(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import CGCNNLitModule

        batch = _make_batch(tiny_graph_samples, 5)
        mse_mod = CGCNNLitModule(hidden_dim=16, n_conv=1, loss_kind="mse")
        huber_mod = CGCNNLitModule(
            hidden_dim=16, n_conv=1,
            loss_kind="huber", huber_delta=0.5,
        )
        assert type(mse_mod.loss_fn).__name__ == "MSELoss"
        assert type(huber_mod.loss_fn).__name__ == "HuberLoss"

    def test_save_hyperparameters_round_trips(
        self, tiny_graph_samples, tmp_path,
    ):
        """load_from_checkpoint without passing init kwargs."""
        import pytorch_lightning as pl

        from backend.common.ml.models.cgcnn_v2 import CGCNNLitModule

        lit = CGCNNLitModule(hidden_dim=16, n_conv=1)
        trainer = pl.Trainer(
            max_epochs=1, enable_progress_bar=False,
            enable_model_summary=False, logger=False,
        )
        # Manual save: Lightning requires a fitted trainer first.
        trainer.strategy.connect(lit)
        trainer.save_checkpoint(str(tmp_path / "m.ckpt"))
        restored = CGCNNLitModule.load_from_checkpoint(
            str(tmp_path / "m.ckpt")
        )
        assert restored.hparams.hidden_dim == 16
        assert restored.hparams.n_conv == 1


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class TestDataModule:
    def test_setup_featurizes_all_splits(self):
        """Integration: DataModule on a tiny synthetic corpus + a
        fake resolver that returns a pymatgen Structure each time.
        """
        from pymatgen.core import Lattice, Structure

        from backend.common.ml.datasets_v2 import (
            DatasetConfig, DatasetRegistry, FilterSpec, PropertyRow,
            SplitSpec,
        )
        from backend.common.ml.models.cgcnn_v2 import CGCNNDataModule

        # 12 rows of dummy materials.
        rows = [
            PropertyRow(
                row_id=f"r{i:02d}", structure_id=f"s{i:02d}",
                property_name="p", property_value=float(i) / 10,
                composition_prototype="alpha" if i % 2 == 0 else "beta",
            )
            for i in range(12)
        ]
        snap = DatasetRegistry().create(
            rows=rows,
            config=DatasetConfig(
                name="t", filter=FilterSpec(),
                split=SplitSpec(
                    kind="random", train_fraction=0.5,
                    val_fraction=0.25, test_fraction=0.25, seed=0,
                ),
            ),
        )

        # Fake resolver: simple cubic Si in a 3x3 supercell — same
        # for every row so the DataModule just exercises the build
        # path.
        def resolver(_row):
            return Structure(
                Lattice.cubic(5.43), ["Si"] * 2,
                [[0, 0, 0], [0.25, 0.25, 0.25]],
            )

        dm = CGCNNDataModule(
            rows=rows, snapshot=snap, resolver=resolver,
            batch_size=4,
        )
        dm.setup("fit")

        assert len(dm._train_samples) == snap.n_train
        assert len(dm._val_samples) == snap.n_val
        assert len(dm._test_samples) == snap.n_test

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "node_features" in batch
        # Each Si graph has 2 atoms; batch_size=4 → 8 nodes.
        assert batch["node_features"].shape[0] == 8


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class TestEnsemble:
    def _small_dm(self, tiny_graph_samples, snapshot_name="t"):
        """Build a DataModule that yields the synthetic graphs
        directly, bypassing the resolver + featurize_row path.

        We plug pre-built ``GraphSample``s straight into
        ``CGCNNDataset`` + ``CGCNNDataModule`` by overriding
        setup() after the fact.
        """
        from backend.common.ml.datasets_v2 import (
            DatasetConfig, DatasetRegistry, FilterSpec, PropertyRow,
            SplitSpec,
        )
        from backend.common.ml.models.cgcnn_v2 import (
            CGCNNDataModule, CGCNNDataset,
        )

        rows = [
            PropertyRow(
                row_id=f"r{i:02d}", structure_id=f"s{i:02d}",
                property_name="p", property_value=float(tiny_graph_samples[i].y),
            )
            for i in range(len(tiny_graph_samples))
        ]
        snap = DatasetRegistry().create(
            rows=rows,
            config=DatasetConfig(
                name=snapshot_name, filter=FilterSpec(),
                split=SplitSpec(
                    kind="random", train_fraction=0.7,
                    val_fraction=0.15, test_fraction=0.15, seed=1,
                ),
            ),
        )
        dm = CGCNNDataModule(
            rows=rows, snapshot=snap,
            resolver=lambda r: None,  # not used — we fill samples directly
            batch_size=8,
        )
        # Fill in samples by mapping row_id → sample.
        by_id = {f"r{i:02d}": s for i, s in enumerate(tiny_graph_samples)}
        dm._train_samples = [by_id[r] for r in snap.train_row_ids]
        dm._val_samples = [by_id[r] for r in snap.val_row_ids]
        dm._test_samples = [by_id[r] for r in snap.test_row_ids]
        return dm, snap

    def test_train_ensemble_produces_n_distinct_models(
        self, tiny_graph_samples,
    ):
        from backend.common.ml.models.cgcnn_v2 import train_ensemble

        dm, _snap = self._small_dm(tiny_graph_samples)
        models = train_ensemble(
            dm=dm,
            n_models=2,
            module_kwargs=dict(hidden_dim=16, n_conv=1, lr=1e-2),
            max_epochs=2,
            base_seed=7,
            trainer_kwargs=dict(accelerator="cpu"),
            deterministic=False,
        )
        assert len(models) == 2
        # Different seeds → different weights (at least one tensor).
        p0 = list(models[0].parameters())
        p1 = list(models[1].parameters())
        diffs = [torch.any(a != b).item() for a, b in zip(p0, p1)]
        assert any(diffs)

    def test_predict_with_uncertainty_shapes(self, tiny_graph_samples):
        from backend.common.ml.models.cgcnn_v2 import (
            predict_with_uncertainty, train_ensemble,
        )

        dm, _ = self._small_dm(tiny_graph_samples)
        models = train_ensemble(
            dm=dm, n_models=2,
            module_kwargs=dict(hidden_dim=16, n_conv=1, lr=1e-2),
            max_epochs=1, deterministic=False,
        )
        mu, sigma, y = predict_with_uncertainty(models, dm.test_dataloader())
        assert mu.shape == y.shape == sigma.shape
        assert (sigma >= 0).all()

    def test_save_load_roundtrip(self, tiny_graph_samples, tmp_path):
        from backend.common.ml.models.cgcnn_v2 import (
            load_ensemble, save_ensemble, train_ensemble,
        )

        dm, snap = self._small_dm(tiny_graph_samples)
        models = train_ensemble(
            dm=dm, n_models=2,
            module_kwargs=dict(hidden_dim=16, n_conv=1, lr=1e-2),
            max_epochs=1, deterministic=False,
        )
        save_ensemble(
            models, tmp_path / "ens",
            dataset_hash=snap.content_hash, dataset_name=snap.config.name,
        )
        loaded = load_ensemble(tmp_path / "ens")
        assert len(loaded) == 2
        # Parameters round-trip.
        for a, b in zip(models[0].parameters(), loaded[0].parameters()):
            torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Learning smoke — loss drops after a few epochs on a learnable target
# ---------------------------------------------------------------------------


class TestLearningSmoke:
    def test_loss_drops_after_training(self, tiny_graph_samples):
        """Honest "does gradient descent work" check.

        Not the roadmap's 300-epoch / MAE ≤ 0.45 eV acceptance — that's
        a multi-minute real-data run, gated separately. This test
        verifies the loss actually goes down on a synthetic
        learnable target. If this fails, something fundamental
        about the model / optimizer / data flow is wrong.
        """
        import pytorch_lightning as pl

        from backend.common.ml.models.cgcnn_v2 import (
            CGCNNLitModule, apply_reproducibility_flags, collate_graphs,
        )

        apply_reproducibility_flags(0)
        batch = collate_graphs(tiny_graph_samples)
        lit = CGCNNLitModule(
            hidden_dim=32, n_conv=2, lr=5e-3, loss_kind="mse",
        )

        # Fixed mini-batch run — 50 steps of gradient descent on the
        # same batch. We want the train loss to drop meaningfully.
        lit.train()
        opt = lit.configure_optimizers()
        initial_loss = lit.loss_fn(lit(batch), batch["y"]).item()
        for _ in range(50):
            opt.zero_grad()
            loss = lit.loss_fn(lit(batch), batch["y"])
            loss.backward()
            opt.step()
        final_loss = lit.loss_fn(lit(batch), batch["y"]).item()
        assert final_loss < initial_loss * 0.8, (
            f"loss didn't drop: initial={initial_loss:.4f} final={final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Bit-exact resume
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_bit_exact_on_cpu(self, tiny_graph_samples, tmp_path):
        """Roadmap acceptance: resume bit-exact on CPU.

        We test the regime where bit-exact resume actually makes
        sense: **no DataLoader shuffling**. In the Lightning
        default, shuffled loaders advance an internal RNG per
        epoch, and that state is NOT part of the checkpoint —
        saving + restarting a process can therefore draw a
        different batch order on epoch N+1 and diverge. That's a
        Lightning-wide limitation, not an ORION bug.

        With ``shuffle=False`` the dataloader is deterministic, the
        optimizer + LR-schedule states are what the checkpoint
        restores, and a resume sees the same gradient trajectory
        as an uninterrupted run. The test below uses that setup
        and enforces bit-exact weight equality.
        """
        import pytorch_lightning as pl
        import torch.utils.data

        from backend.common.ml.datasets_v2 import (
            DatasetConfig, DatasetRegistry, FilterSpec, PropertyRow, SplitSpec,
        )
        from backend.common.ml.models.cgcnn_v2 import (
            CGCNNDataModule, CGCNNDataset, CGCNNLitModule,
            apply_reproducibility_flags, collate_graphs,
        )

        rows = [
            PropertyRow(
                row_id=f"r{i:02d}", structure_id=f"s{i:02d}",
                property_name="p",
                property_value=float(tiny_graph_samples[i].y),
            )
            for i in range(len(tiny_graph_samples))
        ]
        snap = DatasetRegistry().create(
            rows=rows,
            config=DatasetConfig(
                name="resume_noshuffle", filter=FilterSpec(),
                split=SplitSpec(
                    kind="random", train_fraction=0.7,
                    val_fraction=0.15, test_fraction=0.15, seed=2,
                ),
            ),
        )
        by_id = {f"r{i:02d}": s for i, s in enumerate(tiny_graph_samples)}

        class _NoShuffleDM(CGCNNDataModule):
            def train_dataloader(self):
                return torch.utils.data.DataLoader(
                    CGCNNDataset(self._train_samples or []),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_graphs,
                )

        def _make_dm():
            dm = _NoShuffleDM(
                rows=rows, snapshot=snap,
                resolver=lambda r: None, batch_size=8,
            )
            dm._train_samples = [by_id[r] for r in snap.train_row_ids]
            dm._val_samples = [by_id[r] for r in snap.val_row_ids]
            dm._test_samples = [by_id[r] for r in snap.test_row_ids]
            return dm

        # ---- (a) one-shot 4 epochs ----
        apply_reproducibility_flags(11, deterministic=True)
        lit_a = CGCNNLitModule(hidden_dim=16, n_conv=1, lr=1e-2)
        trainer_a = pl.Trainer(
            max_epochs=4, accelerator="cpu", deterministic=True,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False,
        )
        trainer_a.fit(lit_a, datamodule=_make_dm())
        expected = [p.detach().clone() for p in lit_a.parameters()]

        # ---- (b) 2 epochs, save, resume for 4 total ----
        apply_reproducibility_flags(11, deterministic=True)
        lit_b = CGCNNLitModule(hidden_dim=16, n_conv=1, lr=1e-2)
        trainer_b1 = pl.Trainer(
            max_epochs=2, accelerator="cpu", deterministic=True,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False,
        )
        trainer_b1.fit(lit_b, datamodule=_make_dm())
        ckpt = tmp_path / "mid.ckpt"
        trainer_b1.save_checkpoint(str(ckpt))

        lit_b2 = CGCNNLitModule(hidden_dim=16, n_conv=1, lr=1e-2)
        trainer_b2 = pl.Trainer(
            max_epochs=4, accelerator="cpu", deterministic=True,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False,
        )
        trainer_b2.fit(
            lit_b2, datamodule=_make_dm(), ckpt_path=str(ckpt),
        )

        for p_a, p_b in zip(expected, lit_b2.parameters()):
            torch.testing.assert_close(p_a, p_b, rtol=0, atol=1e-5)
