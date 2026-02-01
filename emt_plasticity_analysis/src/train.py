import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, roc_curve, auc
#from esda.moran import Moran_Local
#from libpysal.weights import KNN

def compute_local_morans_weights(data, graphnx, temperature=1.0, k=10):
    """
    Compute per-cell weights that downweight cells in highly autocorrelated label regions.
    Uses esda.Moran_Local. Only uses training nodes to avoid test label leakage.
    """
    train_mask = data['train_mask'].cpu().numpy()
    y = data.y.cpu().numpy()
    # get coordinates from graph nodes
    coords = np.array([[graphnx.nodes[node]['array_row'], graphnx.nodes[node]['array_col']] 
                       for node in graphnx.nodes()])
    train_coords = coords[train_mask]
    train_y = y[train_mask]
    w = KNN.from_array(train_coords, k=k)
    w.transform = 'r'
    moran_loc = Moran_Local(train_y, w, permutations=0)
    local_I = moran_loc.Is
    sample_weights = np.exp(-local_I / temperature)
    sample_weights = sample_weights / sample_weights.mean()
    return torch.tensor(sample_weights, dtype=torch.float32), local_I, coords

def train_and_evaluate_model(model, data, num_epochs=200, learning_rate=0.01, weights=None, return_lowest_loss=False, use_moran_weights=False, moran_temperature=1.0, graphnx=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    continuous_score = data.continuous_score_bool
    train_mask = data['train_mask']
    test_mask = data['test_mask']
    # compute Moran's I sample weights if requested
    moran_sample_weights, local_I_values, coords = None, None, None
    if use_moran_weights:
        moran_sample_weights, local_I_values, coords = compute_local_morans_weights(data, graphnx, temperature=moran_temperature)
        moran_sample_weights = moran_sample_weights.to(device)
    # setup loss function
    if continuous_score:
        base_criterion = torch.nn.MSELoss(reduction='none')
    else:
        if weights:
            unique, counts = np.unique(data.y[train_mask].cpu().numpy(), return_counts=True)
            weights_gnn = 1.0 / torch.tensor(counts, dtype=torch.float32)
            weights_gnn = (weights_gnn / weights_gnn.sum()).to(device)
            base_criterion = torch.nn.NLLLoss(weight=weights_gnn, reduction='none').to(device)
        else:
            base_criterion = torch.nn.NLLLoss(reduction='none').to(device)
    best_performance = float('inf') if continuous_score else 0
    predicted_classes_best_model = None
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)[train_mask]
        if continuous_score:
            per_sample_loss = base_criterion(output, data.y[train_mask].float().squeeze())
        else:
            per_sample_loss = base_criterion(output, data.y[train_mask])
        if use_moran_weights and moran_sample_weights is not None:
            print("loss before", per_sample_loss.mean(), "loss after",(moran_sample_weights * per_sample_loss).mean())
            loss = (moran_sample_weights * per_sample_loss).mean()
        else:
            loss = per_sample_loss.mean()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.edge_attr)[test_mask]
                if continuous_score:
                    mse = mean_squared_error(data.y[test_mask].cpu().numpy(), output.cpu().numpy())
                    print(f"Epoch: {epoch}, MSE: {mse}")
                    performance = mse
                else:
                    _, predicted_classes = torch.max(output, dim=1)
                    f1 = f1_score(data.y[test_mask].cpu().numpy(), predicted_classes.cpu().numpy(), average="weighted")
                    print(f"Epoch: {epoch}, F1 Score: {f1}")
                    probabilities = torch.exp(output)
                    if data.num_classes == 2:
                        performance = roc_auc_score(data.y[test_mask].cpu().numpy(), probabilities[:,1].cpu().numpy())
                    else:
                        performance = roc_auc_score(data.y[test_mask].cpu().numpy(), probabilities.cpu().numpy(), multi_class='ovr')
                    print(f"Epoch: {epoch}, ROC AUC: {performance}")
                if (continuous_score and performance < best_performance) or (not continuous_score and performance > best_performance):
                    best_performance = performance
                    predicted_classes_best_model = output
        if return_lowest_loss:
            predicted_classes_best_model = output
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve{" (Moran-weighted, τ=" + str(moran_temperature) + ")" if use_moran_weights else ""}')
    plt.legend()
    plt.show()
    if use_moran_weights and local_I_values is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(local_I_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Local Moran's I")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Local Moran's I (Training Cells)")
        axes[0].axvline(0, color='red', linestyle='--')
        axes[1].hist(moran_sample_weights.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel("Sample Weight")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Moran-based Sample Weights (τ={moran_temperature})")
        axes[1].axvline(1.0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()
    return best_performance, model, predicted_classes_best_model


def cross_validation_training(folds, graphnx_list, learning_rate=0.009, num_epochs=500, use_moran_weights=False, moran_temperature=1.0):
    all_predicted = []
    all_true = []
    models = []
    for fold_idx, fold_data in enumerate(folds):
        print(f"Training on fold {fold_idx + 1}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from src.models import Net
        model = Net(fold_data, num_features=int(fold_data.num_features), hidden_dim1=16, hidden_dim2=16, dropout_rate=0.58).to(device)
        graphnx = graphnx_list[fold_idx] if isinstance(graphnx_list, list) else graphnx_list
        best_perf, best_model, predicted_fold = train_and_evaluate_model(
            model, fold_data, learning_rate=learning_rate, num_epochs=num_epochs, weights=True, 
            return_lowest_loss=True, use_moran_weights=use_moran_weights, moran_temperature=moran_temperature, graphnx=graphnx)
        print(f"Fold {fold_idx + 1} AUC: {best_perf:.4f}")
        all_predicted.append(predicted_fold.cpu().numpy())
        all_true.append(fold_data.y[fold_data['test_mask']].cpu().numpy())
        models.append(best_model)
    return all_predicted, all_true, models