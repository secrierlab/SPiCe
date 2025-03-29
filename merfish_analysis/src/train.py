import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, roc_curve, auc, roc_auc_score

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
            unique, counts = np.unique(data.y[data.train_mask].cpu().numpy(), return_counts=True)
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
        if data.edge_attr is not None:
            output = model(data.x, data.edge_index)[data.train_mask]
        else:
            output = model(data.x, data.edge_index, data.edge_attr)[data.train_mask]
        loss = criterion(output, data.y[data.train_mask].float().squeeze()) if continuous_score else criterion(output, data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            model.eval()
            with torch.no_grad():
                if data.edge_attr is not None:
                    output = model(data.x, data.edge_index)[data.test_mask]
                else:
                    output = model(data.x, data.edge_index, data.edge_attr)[data.test_mask]
                if continuous_score:
                    mse = mean_squared_error(data.y[data.test_mask].cpu().numpy(), output.cpu().numpy())
                    print(f"Epoch: {epoch}, MSE: {mse}")
                    performance = mse
                else:
                    probabilities = torch.exp(output)
                    _, predicted_classes = torch.max(output, dim=1)
                    f1 = f1_score(data.y[data.test_mask].cpu().numpy(), predicted_classes.cpu().numpy(), average="weighted")
                    print(f"Epoch: {epoch}, F1 Score: {f1}")
                    if data.num_classes == 2:
                        performance = roc_auc_score(data.y[data.test_mask].cpu().numpy(), probabilities[:, 1].cpu().numpy())
                    else:
                        performance = roc_auc_score(data.y[data.test_mask].cpu().numpy(), probabilities.cpu().numpy(), multi_class='ovr')
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

def binary_auc(predicted_probs_np, ground_truth, data, file_save_name=None):
    positive_class_probs = predicted_probs_np[:, 1]
    fpr, tpr, _ = roc_curve(ground_truth, positive_class_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if file_save_name:
        plt.savefig(file_save_name)
    return roc_auc
