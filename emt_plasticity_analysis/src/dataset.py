import numpy as np
import torch
import random
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import KFold

class XeniumSpatialPyg(InMemoryDataset):
    def __init__(self, graphnx, xenium_data, edge_index=None, edge_lengths=None,
                 inductive_split=True, test_cnv_prediction=False, continuous_y=False,
                 tme_and_cnv=False, fold_idx=1, num_folds=None):
        super(XeniumSpatialPyg, self).__init__('.', None, None, None)
        self.graphnx = graphnx
        self.xenium_data = xenium_data
        self.edge_lengths = edge_lengths
        self.inductive_split = inductive_split
        self.test_cnv_prediction = test_cnv_prediction
        self.continuous_y = continuous_y
        self.fold_idx = fold_idx
        self.tme_and_cnv = tme_and_cnv
        self.edge_index = edge_index
        self.num_folds = num_folds
        self.batch_index_to_test = None
        self.process_data()

    def _download(self):
        return

    def create_labels_and_features_for_gnn(self):
        if not self.continuous_y:
            y_gnn_labels = np.array([data['labels'] for node, data in self.graphnx.nodes(data=True)])
            y_gnn_labels[np.isnan(y_gnn_labels)] = -1
            y = torch.tensor(y_gnn_labels, dtype=torch.long)
            data_y = y.clone().detach()
        else:
            y_gnn_labels = np.array([data['EMT_hallmarks'] for node, data in self.graphnx.nodes(data=True)])
            y_gnn_labels = np.round(y_gnn_labels, 5)
            y_gnn_labels[np.isnan(y_gnn_labels)] = -1
            y = torch.tensor(y_gnn_labels, dtype=torch.float)
            data_y = y.clone().detach()
        num_nodes = len(self.graphnx.nodes())
        pca_columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']

        num_genes=len(pca_columns)
        if not self.test_cnv_prediction:
            x_labels = np.array([data['one_hot_Cell_Type'] for node, data in self.graphnx.nodes(data=True)])
            data_x = torch.tensor(x_labels, dtype=torch.float)
        elif self.test_cnv_prediction and self.tme_and_cnv:
            gene_features = np.full((num_nodes, num_genes), np.nan)
            for i, gene in enumerate(pca_columns):
                for j, (node, data) in enumerate(self.graphnx.nodes(data=True)):
                    gene_features[j, i] = data.get(gene, np.nan)
            one_hot_labels = np.array([data['one_hot_Cell_Type'] for node, data in self.graphnx.nodes(data=True)])
            gene_features = np.column_stack((gene_features, one_hot_labels))
            data_x = torch.tensor(gene_features, dtype=torch.float)
        elif self.test_cnv_prediction and not self.tme_and_cnv:
            num_genes = self.xenium_data.shape[1]
            gene_features = np.full((num_nodes, num_genes), np.nan)
            for i in range(num_genes):
                for j, (node, data) in enumerate(self.graphnx.nodes(data=True)):
                    gene_features[j, i] = data.get(list(self.xenium_data.columns)[i], np.nan)
            data_x = torch.tensor(gene_features, dtype=torch.float)
        return data_y, data_x

    def calculate_train_test_from_index_list(self, graphnx, batch_index_to_test, inductive=True):
        if inductive:
            batch_index_to_test_set = set(batch_index_to_test)
            filtered_test = {node for node, data in graphnx.nodes(data=True) if data['sample'] in batch_index_to_test_set}
            filtered_train = {node for node, data in graphnx.nodes(data=True) if data['sample'] not in batch_index_to_test_set}
            node_list = list(graphnx.nodes)
            train_mask = torch.tensor([node in filtered_train for node in node_list], dtype=torch.bool)
            test_mask = torch.tensor([node in filtered_test for node in node_list], dtype=torch.bool)
            return train_mask, test_mask
        else:
            node_list = list(graphnx.nodes)
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
            folds = list(kf.split(node_list))
            fold_index = self.fold_idx - 1
            train_idx, test_idx = folds[fold_index]
            train_mask = torch.tensor([i in train_idx for i in range(len(node_list))], dtype=torch.bool)
            test_mask = torch.tensor([i in test_idx for i in range(len(node_list))], dtype=torch.bool)
            return train_mask, test_mask

    def process_data(self):
        data_y, data_x = self.create_labels_and_features_for_gnn()
        if self.edge_lengths is not None:
            data = Data(edge_index=self.edge_index, edge_attr=self.edge_lengths)
        else:
            data = Data(edge_index=self.edge_index)
        data.num_nodes = self.graphnx.number_of_nodes()
        data.cell_id = np.array([data['cell_id'] for node, data in self.graphnx.nodes(data=True)])
        data.x = data_x
        data.y = data_y
        data.samples = np.array([data['sample'] for node, data in self.graphnx.nodes(data=True)])
        data.continuous_score_bool = True if self.continuous_y else False
        if self.inductive_split:
            samples = np.array([data['sample'] for node, data in self.graphnx.nodes(data=True)])
            unique_samples = np.unique(samples)
            unique_samples = unique_samples[~np.isnan(unique_samples)]
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
            folds = list(kf.split(unique_samples))
            fold_index = self.fold_idx - 1
            train_samples, test_samples = folds[fold_index]
            batch_index_to_test = unique_samples[test_samples]
            self.batch_index_to_test = batch_index_to_test
        train_mask, test_mask = self.calculate_train_test_from_index_list(self.graphnx, self.batch_index_to_test, self.inductive_split)
        if not self.test_cnv_prediction:
            nan_mask = data.y == -1
        else:
            nan_mask = torch.isnan(data.x).any(dim=1)
            nan_mask = torch.logical_or(nan_mask, data.y == -1)
        data['train_mask'] = torch.logical_and(train_mask, ~nan_mask)
        data['test_mask'] = torch.logical_and(test_mask, ~nan_mask)
        if not self.continuous_y:
            data.num_classes = data.y[data['train_mask']].unique().shape[0]
        else:
            data.num_classes = 1
        self.data, self.slices = self.collate([data])
