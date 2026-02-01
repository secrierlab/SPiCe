import torch
import numpy as np
import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, f1_score

class Net(torch.nn.Module):
    def __init__(self, data, num_features, hidden_dim1=16, hidden_dim2=32, dropout_rate=0.5):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, data.num_classes)
        self.dropout_rate = dropout_rate
        self.continuous_score = data.continuous_score_bool

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        if self.continuous_score:
            return x.squeeze()
        else:
            return F.log_softmax(x, dim=1)

def train_single_fold(model, data, num_epochs, learning_rate, device):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_mask = data['train_mask']
    test_mask = data['test_mask']
    continuous_score = data.continuous_score_bool
    if continuous_score:
        base_criterion = torch.nn.MSELoss(reduction='none')
    else:
        unique, counts = np.unique(data.y[train_mask].cpu().numpy(), return_counts=True)
        weights_gnn = 1.0 / torch.tensor(counts, dtype=torch.float32)
        weights_gnn = (weights_gnn / weights_gnn.sum()).to(device)
        base_criterion = torch.nn.NLLLoss(weight=weights_gnn, reduction='none').to(device)
    best_auc = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)[train_mask]
        if continuous_score:
            per_sample_loss = base_criterion(output, data.y[train_mask].float().squeeze())
        else:
            per_sample_loss = base_criterion(output, data.y[train_mask])
        loss = per_sample_loss.mean()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.edge_attr)[test_mask]
                if not continuous_score:
                    probabilities = torch.exp(output)
                    if data.num_classes == 2:
                        auc = roc_auc_score(data.y[test_mask].cpu().numpy(), probabilities[:,1].cpu().numpy())
                    else:
                        auc = roc_auc_score(data.y[test_mask].cpu().numpy(), probabilities.cpu().numpy(), multi_class='ovr')
                    if auc > best_auc:
                        best_auc = auc
    return best_auc

def objective(trial, folds, graphnx):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    hidden_dim1 = trial.suggest_categorical('hidden_dim1', [8, 16, 32, 64])
    hidden_dim2 = trial.suggest_categorical('hidden_dim2', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    num_epochs = trial.suggest_categorical('num_epochs', [200, 300, 500])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_aucs = []
    for fold_idx, fold_data in enumerate(folds):
        moran_weights = None
        model = Net(fold_data, num_features=int(fold_data.num_features),
                   hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout)
        auc = train_single_fold(model, fold_data, num_epochs, lr, device)
        fold_aucs.append(auc)
        trial.report(np.mean(fold_aucs), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(fold_aucs)

def run_hyperparameter_search(folds, graphnx, n_trials=100, study_name='gnn_hyperparam'):
    study = optuna.create_study(direction='maximize', study_name=study_name,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(lambda trial: objective(trial, folds, graphnx), n_trials=n_trials, show_progress_bar=True)
    print("\n" + "="*60)
    print("BEST TRIAL:")
    print("="*60)
    print(f"  AUC: {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # show top 5 trials
    print("\n" + "="*60)
    print("TOP 5 TRIALS:")
    print("="*60)
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['state'] == 'COMPLETE'].sort_values('value', ascending=False).head(5)
    print(trials_df[['value', 'params_lr', 'params_hidden_dim1', 'params_hidden_dim2', 
                     'params_dropout', 'params_use_moran', 'params_moran_temp']].to_string())
    return study

