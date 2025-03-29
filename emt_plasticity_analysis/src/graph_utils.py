import numpy as np
import networkx as nx
import torch
from sklearn.neighbors import NearestNeighbors

def build_graph(xenium_labels, neighbour_no=10):
    G = nx.Graph()
    coords = xenium_labels[['array_row', 'array_col']].values
    nbrs = NearestNeighbors(n_neighbors=neighbour_no, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    epsilon = 1e-6
    for i in range(len(coords)):
        node_id = xenium_labels.index[i]
        G.add_node(node_id, **xenium_labels.iloc[i].to_dict())
        for j in range(1, neighbour_no):  # avoid self-loop
            neighbor_idx = indices[i, j]
            neighbor_node_id = xenium_labels.index[neighbor_idx]
            distance = distances[i, j]
            edge_weight = 1 / (distance + epsilon)
            G.add_edge(node_id, neighbor_node_id, weight=edge_weight)
    return G

def assign_one_hot_celltype(G, attribute='celltype_minor'):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    unique_labels = list(set(nx.get_node_attributes(G, attribute).values()))
    lb.fit(unique_labels)
    for node in G.nodes():
        cell_type = G.nodes[node][attribute]
        one_hot = lb.transform([cell_type])[0]
        G.nodes[node]['one_hot_Cell_Type'] = one_hot
    return lb

def block_split(coordinates, n_blocks):
    coordinates = np.array(coordinates, dtype=float)
    min_lat, max_lat = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
    min_lon, max_lon = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    lat_bins = np.linspace(min_lat, max_lat, n_blocks + 1)
    lon_bins = np.linspace(min_lon, max_lon, n_blocks + 1)
    lat_indices = np.digitize(coordinates[:, 0], bins=lat_bins) - 1
    lon_indices = np.digitize(coordinates[:, 1], bins=lon_bins) - 1
    lat_indices[lat_indices == n_blocks] = n_blocks - 1
    lon_indices[lon_indices == n_blocks] = n_blocks - 1
    blocks = lat_indices * n_blocks + lon_indices
    return blocks

def assign_spatial_blocks(G, n_blocks=4):
    # Assign spatial "sample" labels only to nodes with EMT labels
    emt_nodes = [node for node in G.nodes() if not np.isnan(G.nodes[node]['labels'])]
    coordinates = np.column_stack([
        np.array([G.nodes[node]['array_row'] for node in emt_nodes]),
        np.array([G.nodes[node]['array_col'] for node in emt_nodes])
    ])
    block_labels = block_split(coordinates, n_blocks)
    for idx, node in enumerate(emt_nodes):
        G.nodes[node]['sample'] = block_labels[idx]
    # For nodes without a 'sample' attribute, assign NaN
    for node in G.nodes():
        if 'sample' not in G.nodes[node]:
            G.nodes[node]['sample'] = np.nan
    return

def create_edge_index(graphnx, edge_lengths=False):
    adj = nx.to_scipy_sparse_array(graphnx, nodelist=list(graphnx.nodes())).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    if edge_lengths:
        edge_weights = torch.from_numpy(adj.data).to(torch.float)
        return edge_index, edge_weights
    return edge_index, None
