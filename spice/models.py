"""Graph neural network architectures for spatial cell-state prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpatialGCN(torch.nn.Module):
    """Three-layer graph convolutional network for node classification or regression.

    The architecture follows the SPiCe framework: two hidden GCN layers
    with ReLU activation and dropout, followed by an output GCN layer that
    produces either class log-probabilities or a continuous prediction.

    Parameters
    ----------
    num_features
        Dimensionality of the input node features.
    num_classes
        Number of output classes (set to ``1`` for regression).
    hidden_dim1
        Number of units in the first hidden layer.
    hidden_dim2
        Number of units in the second hidden layer.
    dropout
        Dropout probability applied after each hidden layer.
    continuous
        If ``True``, the model outputs a scalar per node (regression).
        If ``False``, it outputs log-softmax probabilities (classification).
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim1: int = 16,
        hidden_dim2: int = 32,
        dropout: float = 0.5,
        continuous: bool = False,
    ):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, num_classes)
        self.dropout = dropout
        self.continuous = continuous

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Node feature matrix of shape ``(n_nodes, num_features)``.
        edge_index
            COO-format edge index of shape ``(2, n_edges)``.
        edge_attr
            Optional edge weight tensor of shape ``(n_edges,)``.
        """
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)

        if self.continuous:
            return x.squeeze(-1)
        return F.log_softmax(x, dim=1)

    @classmethod
    def from_data(
        cls,
        data,
        hidden_dim1: int = 16,
        hidden_dim2: int = 32,
        dropout: float = 0.5,
    ) -> "SpatialGCN":
        """Convenience constructor that reads feature/class dimensions from a
        PyG :class:`~torch_geometric.data.Data` object.

        Parameters
        ----------
        data
            A PyG Data object (e.g. from :class:`~spice.dataset.SpatialGNNDataset`).
        hidden_dim1, hidden_dim2, dropout
            Architecture hyper-parameters.
        """
        continuous = getattr(data, "continuous_score_bool", False)
        return cls(
            num_features=int(data.num_features),
            num_classes=int(data.num_classes),
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout=dropout,
            continuous=continuous,
        )
