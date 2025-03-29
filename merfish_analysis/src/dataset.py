import os
import random
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data

class MouseSpatialPyg(InMemoryDataset):
    def __init__(self, graphnx, merfish_data, edge_lengths=False, inductive_split=True):
        super(MouseSpatialPyg, self).__init__('.', None, None, None)
        self.graphnx = graphnx
        self.merfish_data = merfish_data
        self.edge_lengths = edge_lengths
        self.inductive_split = inductive_split
        self.process_data()

    def _download(self):
        # Not needed since data is provided locally
        return

    def create_edge_index(self):
        # Convert the NetworkX graph to a SciPy sparse adjacency matrix
        adj = nx.to_scipy_sparse_array(self.graphnx, nodelist=list(self.graphnx.nodes())).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        if self.edge_lengths:
            edge_weights = torch.from_numpy(adj.data).to(torch.float)
            return edge_index, edge_weights
        return edge_index, None

    def create_labels_and_features_for_gnn(self):
        y_gnn_labels = np.array([data['Closest_cell_type_label'] for node, data in self.graphnx.nodes(data=True)])
        x_labels = np.array([data['one_hot_Cell_Type'] for node, data in self.graphnx.nodes(data=True)])
        data_x = torch.tensor(x_labels, dtype=torch.float)
        return y_gnn_labels, data_x

    def calculate_train_test_from_index_list(self, graphnx, batch_index_to_test, inductive=True):
        node_list = list(graphnx.nodes)
        if inductive:
            filtered_test = self.merfish_data[self.merfish_data['Sample'].isin(batch_index_to_test)]
            filtered_train = self.merfish_data[~self.merfish_data['Sample'].isin(batch_index_to_test)]
            filtered_train_set = set(filtered_train.index)
            filtered_test_set = set(filtered_test.index)
            train_mask = torch.tensor([node in filtered_train_set for node in node_list], dtype=torch.bool)
            test_mask = torch.tensor([node in filtered_test_set for node in node_list], dtype=torch.bool)
            return train_mask, test_mask
        else:
            train_mask = torch.tensor([random.random() > 0.25 for _ in range(len(graphnx.nodes))], dtype=torch.bool)
            test_mask = ~train_mask
            return train_mask, test_mask

    def process_data(self):
        y_gnn_labels, data_x = self.create_labels_and_features_for_gnn()
        edge_index, edge_attr = self.create_edge_index()
        data = Data(edge_index=edge_index, x=data_x)
        data.edge_attr = edge_attr
        data.num_nodes = self.graphnx.number_of_nodes()
        y = torch.tensor(y_gnn_labels, dtype=torch.long)
        data.y = y.clone().detach()
        data.continuous_score_bool = False
        data.num_classes = data.y.unique().shape[0]
        batch_indices = self.merfish_data['Sample'].unique()
        batch_index_to_test = np.random.choice(batch_indices, int(0.25 * len(batch_indices)), replace=False)
        self.batch_index_to_test = batch_index_to_test
        train_mask, test_mask = self.calculate_train_test_from_index_list(self.graphnx, self.batch_index_to_test, self.inductive_split)
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        self.data, self.slices = self.collate([data])
