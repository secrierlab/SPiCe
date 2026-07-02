"""Internal training loop for a single model on one fold."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score

from spice.models import SpatialGCN


def train_model(
    model: SpatialGCN,
    data,
    *,
    num_epochs: int = 500,
    learning_rate: float = 0.01,
    class_weights: bool = True,
    device: str | torch.device | None = None,
    verbose: bool = True,
    log_interval: int = 100,
    return_last: bool = True,
) -> dict:
    """Train a single GNN model on one fold of data.

    Parameters
    ----------
    return_last
        If ``True`` (default, matching original implementation), always
        return the **last** evaluated epoch's predictions.  If ``False``,
        return the predictions from the epoch with the best test metric.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    continuous = getattr(data, "continuous_score_bool", False)
    train_mask = data.train_mask
    test_mask = data.test_mask

    if continuous:
        criterion = torch.nn.MSELoss(reduction="none")
    else:
        if class_weights:
            unique, counts = np.unique(data.y[train_mask].cpu().numpy(), return_counts=True)
            w = 1.0 / torch.tensor(counts, dtype=torch.float32)
            w = (w / w.sum()).to(device)
            criterion = torch.nn.NLLLoss(weight=w, reduction="none").to(device)
        else:
            criterion = torch.nn.NLLLoss(reduction="none").to(device)

    best_perf = float("inf") if continuous else 0.0
    best_preds = None
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index, data.edge_attr)[train_mask]

        if continuous:
            loss = criterion(out, data.y[train_mask].float().squeeze()).mean()
        else:
            loss = criterion(out, data.y[train_mask]).mean()

        loss.backward()
        optimizer.step()
        # Keep the loss on-device and defer the host sync: `.item()` every
        # epoch forces the GPU queue to flush at that point. Stacking and
        # transferring once after the loop gives the identical float values
        # without `num_epochs` separate synchronisations.
        loss_history.append(loss.detach())

        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                out_test = model(data.x, data.edge_index, data.edge_attr)[test_mask]
                if continuous:
                    perf = mean_squared_error(
                        data.y[test_mask].cpu().numpy(), out_test.cpu().numpy(),
                    )
                    improved = perf < best_perf
                else:
                    probs = torch.exp(out_test)
                    if data.num_classes == 2:
                        perf = roc_auc_score(
                            data.y[test_mask].cpu().numpy(),
                            probs[:, 1].cpu().numpy(),
                        )
                    else:
                        perf = roc_auc_score(
                            data.y[test_mask].cpu().numpy(),
                            probs.cpu().numpy(),
                            multi_class="ovr",
                        )
                    improved = perf > best_perf

                if improved:
                    best_perf = perf
                if improved and not return_last:
                    best_preds = out_test.clone()

                # When return_last=True, always overwrite with the
                # current epoch's output (matches original behaviour).
                if return_last:
                    best_preds = out_test.clone()

                if verbose:
                    metric = "MSE" if continuous else "AUC"
                    print(
                        f"  Epoch {epoch:>4d} | Loss {loss.item():.4f} | "
                        f"Test {metric}: {perf:.4f}"
                    )

    return {
        "best_performance": best_perf,
        "model": model,
        "predictions": best_preds,
        "loss_history": torch.stack(loss_history).cpu().tolist(),
    }
