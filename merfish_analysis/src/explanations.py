# src/explanations.py

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from captum.attr import IntegratedGradients
from torch_geometric.nn import to_captum_model, to_captum_input
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from torch_geometric.explain import Explainer, GNNExplainer

def generate_node_explanations(model, data, test_labels_index, no_explan=50, mask_type="node", label_list=None):
    """
    Generate node explanations using Integrated Gradients.

    Parameters:
      model: trained model.
      data: PyG Data object.
      test_labels_index: DataFrame with columns 'Indices' and 'Label' for test nodes.
      no_explan: number of explanations per label.
      mask_type: type of mask to use (default "node").
      label_list: Optional list to map index to label names.
    
    Returns:
      score_df_intgrad_aggregated: Dictionary of DataFrames for each label containing aggregated importance scores.
      p_values_df: DataFrame of p-values from the Mann-Whitney U test.
    """
    score_df_intgrad_aggregated = {}
    unique_labels = test_labels_index['Label'].unique()
    
    for label in unique_labels:
        print(f"Processing node explanations for label: {label}")
        indices = test_labels_index[test_labels_index['Label'] == label]['Indices'].values
        indices_random = np.random.choice(indices, no_explan)
        aggregated_df_intgrad = pd.DataFrame()
        for index_val in tqdm(indices_random, desc=f"Nodes for label {label}"):
            index_val = int(index_val)
            if int(data.y[index_val]) == label:
                captum_model = to_captum_model(model, mask_type, index_val)
                inputs, additional_forward_args = to_captum_input(data.x, data.edge_index, mask_type, data.edge_attr)
                ig = IntegratedGradients(captum_model)
                ig_attr = ig.attribute(inputs=inputs, target=int(data.y[index_val]),
                                       additional_forward_args=additional_forward_args,
                                       internal_batch_size=1)
                ig_attr_tensor = ig_attr[0]
                ig_attr_numpy = ig_attr_tensor.squeeze().detach().cpu().numpy()
                ig_attr_df = pd.DataFrame(ig_attr_numpy)
                ig_attr_df_sum = pd.DataFrame(ig_attr_df.mean(axis=0)).T
                aggregated_df_intgrad = pd.concat([aggregated_df_intgrad, ig_attr_df_sum], ignore_index=True)
        score_df_intgrad_aggregated[label] = aggregated_df_intgrad

    # Generate permutation aggregated scores
    permuted_aggregated = pd.DataFrame()
    indices = test_labels_index['Indices'].values
    indices_random = np.random.choice(indices, no_explan)
    for index_val in tqdm(indices_random, desc="Permutation node explanations"):
        index_val = int(index_val)
        captum_model = to_captum_model(model, mask_type, index_val)
        inputs, additional_forward_args = to_captum_input(data.x, data.edge_index, mask_type, data.edge_attr)
        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(inputs=inputs, target=int(data.y[index_val]),
                               additional_forward_args=additional_forward_args,
                               internal_batch_size=1)
        ig_attr_tensor = ig_attr[0]
        ig_attr_numpy = ig_attr_tensor.squeeze().detach().cpu().numpy()
        ig_attr_df = pd.DataFrame(ig_attr_numpy)
        ig_attr_df_sum = pd.DataFrame(ig_attr_df.mean(axis=0)).T
        permuted_aggregated = pd.concat([permuted_aggregated, ig_attr_df_sum], ignore_index=True)

    if label_list is not None:
        index_to_label_mapping = {i: label for i, label in enumerate(label_list)}
        permuted_aggregated = permuted_aggregated.rename(columns=index_to_label_mapping)
        for key in score_df_intgrad_aggregated.keys():
            score_df_intgrad_aggregated[key] = score_df_intgrad_aggregated[key].rename(columns=index_to_label_mapping)

    # Compute p-values using the Mann-Whitney U test
    p_values_df = pd.DataFrame(index=unique_labels, columns=score_df_intgrad_aggregated[unique_labels[0]].columns)
    for label in unique_labels:
        for feature_idx in score_df_intgrad_aggregated[label]:
            real_scores = score_df_intgrad_aggregated[label][feature_idx]
            permuted_scores = permuted_aggregated[feature_idx]
            _, p_value = mannwhitneyu(real_scores, permuted_scores, alternative='greater')
            p_values_df.loc[label, feature_idx] = p_value

    return score_df_intgrad_aggregated, p_values_df


def generate_edge_explanations(model, data, G, no_explan=50, cell_type="default", output_dir="results/"):
    """
    Generate edge explanations using GNNExplainer and compute statistical significance of
    edge importance across cell type pairs.

    Parameters:
      model: trained GNN model.
      data: PyG Data object.
      G: NetworkX graph corresponding to the data (should have 'Cell_Type' attribute).
      no_explan: number of explanations per label group.
      cell_type: string identifier used in output filenames.
      output_dir: directory where output files will be saved.
      
    Returns:
      results: Dictionary mapping each test label to its DataFrame of mean importance and p-values.
    """
    # Initialize the explainer with GNNExplainer algorithm
    
    explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',  # Model returns log probabilities.
    ),
    )

    # Assign index labels to nodes
    for idx, (node, attrs) in enumerate(G.nodes(data=True)):
        G.nodes[node]['index_label'] = idx


    # Step 1: Extract the full dataset indices corresponding to the test mask
    test_indices = torch.where(data['test_mask'])[0]  # Indices of the test nodes

    # Step 2: Separate the test indices by label (0 and 1)
    test_labels = data.y[test_indices]

    print("test_labels")
    print(test_labels)
    label_y = 1
    indices_label_1 = test_indices[(test_labels == label_y).nonzero(as_tuple=True)[0]]
    #print head of indices_label_1
    print(indices_label_1[:5])
    # Step 3: Randomly select 100 nodes from each label category
    random_1_indices = indices_label_1[torch.randint(len(indices_label_1), (no_explan,))]
    indices_permutation= test_indices[torch.randperm(len(test_indices))[:no_explan]]

    # Combine the selected indices for explanation
    selected_indices = torch.cat([indices_permutation, random_1_indices])

    # Step 4: Loop through the selected indices and generate explanations
    edge_explanations = []
    #node_explanations = []

    #tqdm
    print("Generating link explanations...")

    for idx in tqdm(selected_indices):
        explanation = explainer(x=data.x, edge_index=data.edge_index, index=idx.item(),edge_attr=data.edge_attr)
        edge_explanations.append(explanation.edge_mask.cpu())  # Move to CPU
        #node_explanations.append(explanation.node_mask.cpu())  # Move to CPU

    edge_explanations = torch.stack(edge_explanations)
    #save the edge_explanations by converting to numpy
    edge_explanations=edge_explanations.cpu().detach().numpy()

    # Map edge indices to cell type pairs once
    index_label_to_node = {d['index_label']: n for n, d in G.nodes(data=True)}
    cell_type_pairs = []
    print("calculating tuples")
    for src, dst in data.edge_index.T:
        src_node = index_label_to_node[src.item()]
        dst_node = index_label_to_node[dst.item()]
        src_cell_type = G.nodes[src_node]['Cell_Type']
        dst_cell_type = G.nodes[dst_node]['Cell_Type']
        cell_type_pairs.append(tuple(sorted([src_cell_type, dst_cell_type])))

    # Dictionary to store mean importance by cell type pair per explanation
    cell_type_pair_importances = {'permutation': [], 'nonpermutation': []}
    index_categories = ['permutation'] * len(random_1_indices) + ['nonpermutation'] * len(indices_permutation)
    print("calculating mean importance")
    # Aggregate mean importance per cell type pair for each explanation
    for edge_mask, category in zip(edge_explanations, index_categories):
        edge_importance = edge_mask
        # Initialize dictionary to collect importance values for each cell type pair in this explanation
        explanation_importance = {}
        for pair, importance in zip(cell_type_pairs, edge_importance):
            if pair not in explanation_importance:
                explanation_importance[pair] = []
            explanation_importance[pair].append(importance)
        
        # Calculate mean importance for each cell type pair in this explanation
        mean_explanation_importance = {pair: np.mean(values) for pair, values in explanation_importance.items()}
        cell_type_pair_importances[category].append(mean_explanation_importance)
    print("calculating p values_allpairs")
    # Convert data to DataFrame format for significance testing
    data_df = []
    all_pairs = set(pair for explanation in cell_type_pair_importances['permutation'] for pair in explanation)
    all_pairs.update(pair for explanation in cell_type_pair_importances['nonpermutation'] for pair in explanation)
    print("calculating p values")
    for pair in all_pairs:
        perm_values = [explanation.get(pair, 0) for explanation in cell_type_pair_importances['permutation']]
        non_perm_values = [explanation.get(pair, 0) for explanation in cell_type_pair_importances['nonpermutation']]
        data_df.append({
            'Cell_Type_Pair': pair,
            'Non_Permutation_Values': non_perm_values,
            'Permutation_Values': perm_values,
            'Avg_Importance_Non_Permutation': np.mean(non_perm_values),
            'Avg_Importance_Permutation': np.mean(perm_values),
        })

    df_counts = pd.DataFrame(data_df)

    # Perform t-test for each cell type pair
    p_values = []
    for index, row in df_counts.iterrows():
        t_stat, p_value = ttest_ind(row['Non_Permutation_Values'], row['Permutation_Values'], alternative='greater')
        p_values.append(p_value)

    df_counts['p_value'] = p_values

    _, corrected_p_values, _, _ = multipletests(df_counts['p_value'], method='bonferroni')

    #
    safe_cell_type = str(cell_type).replace("/", "_")  # Replace slashes with underscores
    df_counts.to_csv(f"edge_{safe_cell_type}_avg_importance_and_p_values2702.csv", index=False)
