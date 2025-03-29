import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, roc_curve, auc

def train_and_evaluate_model(model, data, num_epochs=200, learning_rate=0.01, weights=None, return_lowest_loss=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    continuous_score = data.continuous_score_bool
    if continuous_score:
        criterion = torch.nn.MSELoss()
    else:
        if weights:
            unique, counts = np.unique(data.y[data['train_mask']].cpu().numpy(), return_counts=True)
            weights_gnn = 1.0 / torch.tensor(counts, dtype=torch.float32)
            weights_gnn = weights_gnn / weights_gnn.sum()
            weights_gnn = weights_gnn.to(device)
            criterion = torch.nn.NLLLoss(weight=weights_gnn).to(device)
        else:
            criterion = torch.nn.NLLLoss().to(device)
    best_performance = float('inf') if continuous_score else 0
    predicted_classes_best_model = None
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)[data['train_mask']]
        loss = criterion(output, data.y[data['train_mask']].float().squeeze()) if continuous_score else criterion(output, data.y[data['train_mask']])
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            model.eval()
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.edge_attr)[data['test_mask']]
                if continuous_score:
                    mse = mean_squared_error(data.y[data['test_mask']].cpu().numpy(), output.cpu().numpy())
                    print(f"Epoch: {epoch}, MSE: {mse}")
                    performance = mse
                else:
                    _, predicted_classes = torch.max(output, dim=1)
                    f1 = f1_score(data.y[data['test_mask']].cpu().numpy(), predicted_classes.cpu().numpy(), average="weighted")
                    print(f"Epoch: {epoch}, F1 Score: {f1}")
                    probabilities = torch.exp(output)
                    if data.num_classes == 2:
                        performance = roc_auc_score(data.y[data['test_mask']].cpu().numpy(), probabilities[:,1].cpu().numpy())
                    else:
                        performance = roc_auc_score(data.y[data['test_mask']].cpu().numpy(), probabilities.cpu().numpy(), multi_class='ovr')
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
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    return best_performance, model, predicted_classes_best_model

def cross_validation_training(folds, learning_rate=0.01, num_epochs=500):
    all_predicted = []
    all_true = []
    models = []
    for fold_idx, fold_data in enumerate(folds):
        print(f"Training on fold {fold_idx + 1}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from src.models import Net
        model = Net(fold_data, num_features=int(fold_data.num_features), hidden_dim1=16, hidden_dim2=32, dropout_rate=0.5).to(device)
        best_perf, best_model, predicted_fold = train_and_evaluate_model(model, fold_data, learning_rate=learning_rate, num_epochs=num_epochs, weights=True, return_lowest_loss=True)
        print(f"Fold {fold_idx + 1} AUC: {best_perf:.4f}")
        all_predicted.append(predicted_fold.cpu().numpy())
        all_true.append(fold_data.y[fold_data['test_mask']].cpu().numpy())
        models.append(best_model)
    return all_predicted, all_true, models
