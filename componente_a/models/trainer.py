"""Loop de entrenamiento compartido para AE, DAE y VAE."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def train_ae(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 64,
    grad_clip_norm: float = 0.0,
    lr_schedule: Optional[str] = None,
    noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    wandb_run=None,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """Entrena model.loss(x_in, x_target) con early stopping en val_loss.

    X_tensor: datos ya normalizados (solo rows normales del split de entrenamiento).
    noise_fn: callable(batch) -> batch_corrupted; usado para DAE.
              El target siempre es el batch limpio.
    lr_schedule: None | 'cosine' | 'plateau'
      - cosine: CosineAnnealingLR, decae lr de lr_max a lr/100 en max_epochs.
      - plateau: ReduceLROnPlateau(factor=0.5, patience=patience//3).
    wandb_run: si está activo, loguea train_loss/val_loss/epoch por epoch.
    """
    model = model.to(device)
    X = X_tensor.to(device)

    n = len(X)
    n_val = max(1, int(n * 0.15))
    n_tr = n - n_val
    # Split temporal: últimas 15% filas como val interno (preserva orden)
    X_tr, X_val = X[:n_tr], X[n_tr:]

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr / 100)
    elif lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                      patience=max(1, patience // 3), min_lr=lr / 100)

    best_val = float("inf")
    patience_left = patience
    best_state: Optional[dict] = None
    history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        epoch_losses = []
        for i in range(0, n_tr, batch_size):
            batch = X_tr[perm[i : i + batch_size]]
            batch_in = noise_fn(batch) if noise_fn is not None else batch
            optimizer.zero_grad()
            loss = model.loss(batch_in, batch)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(X_val, X_val).item()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch,
                        "train_loss": float(np.mean(epoch_losses)),
                        "val_loss": float(val_loss),
                        "lr": current_lr})

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if wandb_run is not None:
            try:
                wandb_run.log({"train_loss": float(np.mean(epoch_losses)),
                               "val_loss": val_loss, "lr": current_lr, "epoch": epoch})
            except Exception:
                pass

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    return history
