import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import NearestNeighbors

def compute_neighborhood_features(X, coords, k=10):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes, n_features = X.shape
    neighbor_mean = np.zeros((n_nodes, n_features))
    for i in range(n_nodes):
        neighbor_idx = indices[i, 1:]  
        neighbor_mean[i] = X[neighbor_idx].sum(axis=0)
    X_augmented = np.hstack([X, neighbor_mean])
    return X_augmented

def get_data_from_fold(fold_data, graphnx):
    train_mask = fold_data['train_mask'].cpu().numpy()
    test_mask = fold_data['test_mask'].cpu().numpy()
    X = fold_data.x.cpu().numpy()
    y = fold_data.y.cpu().numpy()
    coords = np.array([[graphnx.nodes[node]['array_row'], graphnx.nodes[node]['array_col']] 
                       for node in graphnx.nodes()])
    return X, y, coords, train_mask, test_mask

def evaluate_classifier(clf, X_train, y_train, X_test, y_test, n_classes):
    clf.fit(X_train, y_train)
    if n_classes == 2:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    return auc

def run_baseline_comparison(folds, graphnx, k=10):

    results = {
        'RF': [],
        'RF + Neighbours': [],
        'MLP': [],
        'MLP + Neighbours': [],
        'GNN': []
    }
    
    for fold_idx, fold_data in enumerate(folds):
        X, y, coords, train_mask, test_mask = get_data_from_fold(fold_data, graphnx)
        n_classes = len(np.unique(y[train_mask]))
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        coords_train, coords_test = coords[train_mask], coords[test_mask]
        X_aug = compute_neighborhood_features(X, coords, k=k)
        X_train_aug, X_test_aug = X_aug[train_mask], X_aug[test_mask]
                
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        auc = evaluate_classifier(clf, X_train, y_train, X_test, y_test, n_classes)
        results['RF'].append(auc)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        auc = evaluate_classifier(clf, X_train_aug, y_train, X_test_aug, y_test, n_classes)
        results['RF + Neighbours'].append(auc)
        
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        auc = evaluate_classifier(clf, X_train, y_train, X_test, y_test, n_classes)
        results['MLP'].append(auc)
        
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        auc = evaluate_classifier(clf, X_train_aug, y_train, X_test_aug, y_test, n_classes)
        results['MLP + Neighbours'].append(auc)
        
        from src.models import Net
        from src.train import train_and_evaluate_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(fold_data, num_features=int(fold_data.num_features), hidden_dim1=16, hidden_dim2=32, dropout_rate=0.5).to(device)
        auc, _, _ = train_and_evaluate_model(model, fold_data, num_epochs=500, learning_rate=0.01, weights=True, return_lowest_loss=True)
        results['GNN'].append(auc)
    
    print("SUMMARY: Mean AUC ± Std across folds")
    summary = []
    for name, aucs in results.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"  {name:20s}: {mean_auc:.4f} ± {std_auc:.4f}")
        summary.append({'Model': name, 'Mean AUC': mean_auc, 'Std': std_auc})
    
    df_results = pd.DataFrame(summary).sort_values('Mean AUC', ascending=False)
    return df_results, results

def plot_baseline_comparison(results):
    import matplotlib.pyplot as plt
    models = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]
    sorted_idx = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    
    colors = ['#3498db' if 'Neighbor' in m else '#95a5a6' for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, means, xerr=stds, color=colors, edgecolor='black', capsize=5)
    ax.set_xlabel('ROC AUC')
    ax.set_title('GNN vs Baseline Classifiers')
    ax.set_xlim(0.5, 1.0)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{mean:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/baseline_comparison.pdf', bbox_inches='tight')
    plt.show()
