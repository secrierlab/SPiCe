import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, data, num_features, hidden_dim1=50, hidden_dim2=100, dropout_rate=0.5):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, data.num_classes)
        self.dropout_rate = dropout_rate
        self.continuous_score = data.continuous_score_bool

    def forward(self, x, edge_index, edge_attr=None):
        # Use edge_attr only if provided
        if edge_attr is None:
            x = F.relu(self.conv1(x, edge_index))
        else:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if edge_attr is None:
            x = F.relu(self.conv2(x, edge_index))
        else:
            x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if edge_attr is None:
            x = self.conv3(x, edge_index)
        else:
            x = self.conv3(x, edge_index, edge_attr)
        
        if self.continuous_score:
            return x.squeeze()
        else:
            return F.log_softmax(x, dim=1)
