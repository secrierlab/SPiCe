import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, r2_score

def evaluate_classification(all_predicted, all_true, n_classes):
    """
    Evaluates classification performance across folds.
    
    For a two-state (binary) model (n_classes == 2), the function computes the ROC
    curve using the predicted probabilities for the positive class (column 1) and 
    calculates the AUC for each fold.
    
    For multi-class problems, it computes per-class, micro-, and macro-average AUC.
    
    Parameters:
      all_predicted: List of NumPy arrays with predicted probabilities for each fold.
      all_true: List of ground truth labels (as 1D arrays) for each fold.
      n_classes: Number of classes (for binary classification, set n_classes==2).
      
    Returns:
      A DataFrame with the AUC scores (per fold and per class for multi-class, or per fold for binary).
    """
    if n_classes == 2:
        # Binary classification evaluation
        n_folds = len(all_predicted)
        cumulative_roc_auc = 0.0
        auc_scores = []
        for fold_idx in range(n_folds):
            # Compute ROC curve using the predicted probability for the positive class.
            fpr, tpr, _ = roc_curve(all_true[fold_idx], all_predicted[fold_idx][:, 1])
            roc_auc = auc(fpr, tpr)
            cumulative_roc_auc += roc_auc
            auc_scores.append({
                'Fold': fold_idx + 1,  # 1-based fold index
                'AUC': roc_auc
            })
        average_roc_auc = cumulative_roc_auc / n_folds
        print("AUC for each fold:")
        for fold_score in auc_scores:
            print(f"Fold {fold_score['Fold']}: {fold_score['AUC']:.4f}")
        print(f"Average AUC across all folds: {average_roc_auc:.4f}")
        df_auc = pd.DataFrame(auc_scores)
        return df_auc
    else:
        # Multi-class evaluation
        n_folds = len(all_predicted)
        cumulative_roc_auc = np.zeros(n_classes)
        cumulative_roc_auc_micro = 0.0
        cumulative_roc_auc_macro = 0.0
        auc_scores = []
        from sklearn.preprocessing import label_binarize
        for fold_idx in range(n_folds):
            true_binarized = label_binarize(all_true[fold_idx], classes=list(range(n_classes)))
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(true_binarized[:, i], all_predicted[fold_idx][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                cumulative_roc_auc[i] += roc_auc[i]
                auc_scores.append({'Fold': fold_idx + 1, 'Class': i, 'AUC': roc_auc[i]})
            fpr["micro"], tpr["micro"], _ = roc_curve(true_binarized.ravel(), all_predicted[fold_idx].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            cumulative_roc_auc_micro += roc_auc["micro"]
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            cumulative_roc_auc_macro += roc_auc["macro"]
        avg_auc_per_class = cumulative_roc_auc / n_folds
        avg_auc_micro = cumulative_roc_auc_micro / n_folds
        avg_auc_macro = cumulative_roc_auc_macro / n_folds
        print("Average AUC per class:")
        for i in range(n_classes):
            print(f"Class {i}: {avg_auc_per_class[i]:.4f}")
        print(f"Average Micro-average AUC: {avg_auc_micro:.4f}")
        print(f"Average Macro-average AUC: {avg_auc_macro:.4f}")
        df_auc = pd.DataFrame(auc_scores)
        return df_auc

def evaluate_regression(all_predicted, all_true):
    n_folds = len(all_predicted)
    cumulative_r2 = 0.0
    cumulative_corr = 0.0
    eval_scores = []
    for fold_idx in range(n_folds):
        y_true = all_true[fold_idx]
        y_pred = all_predicted[fold_idx]
        r2 = r2_score(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred.squeeze())[0, 1]
        cumulative_r2 += r2
        cumulative_corr += corr
        eval_scores.append({'Fold': fold_idx + 1, 'R-squared': r2, 'Correlation': corr})
    avg_r2 = cumulative_r2 / n_folds
    avg_corr = cumulative_corr / n_folds
    print("R-squared and Correlation per fold:")
    for res in eval_scores:
        print(f"Fold {res['Fold']}: R-squared: {res['R-squared']:.4f}, Correlation: {res['Correlation']:.4f}")
    print(f"Average R-squared: {avg_r2:.4f}")
    print(f"Average Correlation: {avg_corr:.4f}")
    df_eval = pd.DataFrame(eval_scores)
    return df_eval
