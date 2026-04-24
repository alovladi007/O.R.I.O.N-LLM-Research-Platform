"""CGCNN-style crystal graph network for Session 6.4.

The architecture follows Xie & Grossman (2018) — Crystal Graph
Convolutional Neural Network — at a minimal level:

1. Embed node features via a linear layer.
2. Apply ``n_conv`` CGCNN message-passing convolutions.
3. Mean-pool node embeddings across each crystal.
4. MLP on the pooled vector → 1 regression output.

We deliberately keep this **PyG-free** and write the message-passing
kernel as plain ``index_add_`` / gather ops. The reason:

- ``features_v2.build_radius_graph`` already returns ``edge_index`` +
  ``edge_features`` as numpy arrays; wrapping them in a PyG
  ``Data`` object adds a dep (torch_geometric + its C++ extensions
  + a 300 MB install) without buying us much — the CGCNN convolution
  is ~15 lines of torch ops either way.
- Keeps Session 6.4 installable on CI without a PyG wheel dance.

A future session that needs heavier GNNs (e.g. equivariant message
passing) can migrate to PyG in one shot.

CGCNN convolution
-----------------

Given node features ``h_i`` (shape ``(n_atoms, n_feat)``) and edge
features ``e_ij`` (shape ``(n_edges, n_edge_feat)``), each
convolution layer computes::

    z_ij   = [h_i || h_j || e_ij]                  (concat)
    g_ij   = sigmoid(W_g z_ij + b_g)               (gate)
    c_ij   = softplus(W_c z_ij + b_c)              (candidate)
    m_i    = sum_j g_ij ⊙ c_ij                     (aggregate over neighbours)
    h_i'   = softplus(h_i + batchnorm(m_i))         (residual)

The gated addition is the "CGCNN conv" block from the paper
(Eq. 5). We use ``softplus`` over ReLU so gradients stay healthy
for negative pre-activations on small corpora.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNConv(nn.Module):
    """One CGCNN message-passing layer.

    Args
    ----
    node_dim
        Dimensionality of node features (unchanged across layers).
    edge_dim
        Dimensionality of edge features (the 10-d vector from
        :func:`backend.common.ml.features_v2.build_radius_graph`).
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.in_dim = 2 * node_dim + edge_dim
        self.gate = nn.Linear(self.in_dim, node_dim)
        self.cand = nn.Linear(self.in_dim, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(
        self,
        h: torch.Tensor,          # (N, F)
        edge_index: torch.Tensor, # (2, E) long
        edge_feat: torch.Tensor,  # (E, F_e)
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]   # (E, F)
        h_dst = h[dst]   # (E, F)
        z = torch.cat([h_src, h_dst, edge_feat], dim=1)   # (E, in_dim)
        g = torch.sigmoid(self.gate(z))                   # (E, F)
        c = F.softplus(self.cand(z))                      # (E, F)
        m = g * c                                         # (E, F)
        # Aggregate messages to the source node (directed pair →
        # both directions are present in ``edge_index`` by
        # build_radius_graph's convention, so sum over src gives
        # the full neighbourhood aggregation).
        agg = torch.zeros_like(h)
        agg.index_add_(0, src, m)
        # Residual update with batch normalization on the
        # aggregated term only (matches the paper).
        agg_bn = self.bn(agg) if h.shape[0] > 1 else agg
        return F.softplus(h + agg_bn)


class CGCNN(nn.Module):
    """Full Session 6.4 CGCNN regressor.

    Input tensors per sample (handled by the DataModule's collate):

    - ``node_features``  shape ``(N_total, F_node)``
    - ``edge_index``     shape ``(2, E_total)``, long
    - ``edge_features``  shape ``(E_total, F_edge)``
    - ``graph_index``    shape ``(N_total,)`` long — which graph
      each atom belongs to (``0..B-1``). Used for mean pooling.

    Output: shape ``(B,)`` — one scalar per graph.
    """

    def __init__(
        self,
        node_feat_dim: int = 35,
        edge_feat_dim: int = 10,
        hidden_dim: int = 64,
        n_conv: int = 3,
        mlp_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.n_conv = n_conv
        self.dropout = dropout

        self.embed = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [CGCNNConv(hidden_dim, edge_feat_dim) for _ in range(n_conv)]
        )
        layers: List[nn.Module] = []
        prev = hidden_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.Softplus())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        graph_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(node_features)
        for conv in self.convs:
            h = conv(h, edge_index, edge_features)
        # Mean-pool by graph.
        B = int(graph_index.max().item()) + 1 if graph_index.numel() > 0 else 0
        if B == 0:
            return torch.zeros(0, device=h.device)
        pooled = torch.zeros(B, h.shape[1], device=h.device, dtype=h.dtype)
        counts = torch.zeros(B, device=h.device, dtype=h.dtype)
        pooled.index_add_(0, graph_index, h)
        counts.index_add_(
            0, graph_index,
            torch.ones(graph_index.shape[0], device=h.device, dtype=h.dtype),
        )
        counts = counts.clamp_min_(1.0)
        pooled = pooled / counts.unsqueeze(1)
        return self.mlp(pooled).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
