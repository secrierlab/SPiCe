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
    score_df_intgrad_aggregated = {}
    unique_labels = test_labels_index['Label'].unique()
    for label in unique_labels:
        print(f"Processing node explanations for label: {label}")
        indices = test_labels_index[test_labels_index['Label'] == label]['Indices'].values
        indices_random = np.random.choice(indices, no_explan)
        aggregated_df = pd.DataFrame()
        for index_val in tqdm(indices_random, desc=f"Nodes for label {label}"):
            index_val = int(index_val)
            if int(data.y[index_val]) == label:
                captum_model = to_captum_model(model, mask_type, index_val)
                inputs, add_args = to_captum_input(data.x, data.edge_index, mask_type, data.edge_attr)
                ig = IntegratedGradients(captum_model)
                ig_attr = ig.attribute(inputs=inputs, target=int(data.y[index_val]),
                                         additional_forward_args=add_args,
                                         internal_batch_size=1)
                ig_attr_tensor = ig_attr[0]
                ig_attr_np = ig_attr_tensor.squeeze().detach().cpu().numpy()
                ig_df = pd.DataFrame(ig_attr_np)
                ig_df_sum = pd.DataFrame(ig_df.mean(axis=0)).T
                aggregated_df = pd.concat([aggregated_df, ig_df_sum], ignore_index=True)
        score_df_intgrad_aggregated[label] = aggregated_df

    permuted_aggregated = pd.DataFrame()
    indices = test_labels_index['Indices'].values
    indices_random = np.random.choice(indices, no_explan)
    for index_val in tqdm(indices_random, desc="Permutation node explanations"):
        index_val = int(index_val)
        captum_model = to_captum_model(model, mask_type, index_val)
        inputs, add_args = to_captum_input(data.x, data.edge_index, mask_type, data.edge_attr)
        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(inputs=inputs, target=int(data.y[index_val]),
                                 additional_forward_args=add_args,
                                 internal_batch_size=1)
        ig_attr_tensor = ig_attr[0]
        ig_attr_np = ig_attr_tensor.squeeze().detach().cpu().numpy()
        ig_df = pd.DataFrame(ig_attr_np)
        ig_df_sum = pd.DataFrame(ig_df.mean(axis=0)).T
        permuted_aggregated = pd.concat([permuted_aggregated, ig_df_sum], ignore_index=True)
    if label_list is not None:
        index_to_label = {i: label for i, label in enumerate(label_list)}
        permuted_aggregated = permuted_aggregated.rename(columns=index_to_label)
        for key in score_df_intgrad_aggregated:
            score_df_intgrad_aggregated[key] = score_df_intgrad_aggregated[key].rename(columns=index_to_label)
    p_values_df = pd.DataFrame(index=unique_labels, columns=score_df_intgrad_aggregated[unique_labels[0]].columns)
    for label in unique_labels:
        for feat in score_df_intgrad_aggregated[label]:
            real_scores = score_df_intgrad_aggregated[label][feat]
            permuted_scores = permuted_aggregated[feat]
            _, p_value = mannwhitneyu(real_scores, permuted_scores, alternative='greater')
            p_values_df.loc[label, feat] = p_value
    #save p-values
    p_values_df.to_csv("results/node_p_values_quick.csv", index=True)
    return score_df_intgrad_aggregated, p_values_df

def generate_edge_explanations(model, data, G, no_explan=50, cell_type="default", output_dir="results/"):
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

    for label_y in test_labels.unique():

        print("explanations for label")
        print(label_y) 
        label_y = label_y.item()
        indices_label_1 = test_indices[(test_labels == label_y).nonzero(as_tuple=True)[0]]
     
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
        edge_explanations_np=edge_explanations.cpu().detach().numpy()

        edge_explanations=edge_explanations_np
        # Map edge indices to cell type pairs once
        index_label_to_node = {d['index_label']: n for n, d in G.nodes(data=True)}
        cell_type_pairs = []
        print("calculating tuples")
        for src, dst in data.edge_index.T:
            src_node = index_label_to_node[src.item()]
            dst_node = index_label_to_node[dst.item()]
            src_cell_type = G.nodes[src_node]['celltype_minor']
            dst_cell_type = G.nodes[dst_node]['celltype_minor']
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

        # Save the results to a CSV file
        df_counts.to_csv("edge_"+str(label_y)+"emt_states_four_states_avg_importance_and_p_values1303.csv", index=False)



