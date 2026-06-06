from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import config as C

def make_optimizer(model, lr=C.LR, weight_decay=0.05):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def cosine_warmup(optimizer, total_steps, warmup_frac=0.1):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)

@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=C.USE_AMP and device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=C.USE_AMP and device.type == "cuda"):
            out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)

def train(model, train_loader, val_loader, device, *, epochs=C.EPOCHS, lr=C.LR,
          label_smoothing=0.1, patience=None, log_every=1):

    C.set_seed()
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = make_optimizer(model, lr=lr)
    total_steps = epochs * max(1, len(train_loader))
    scheduler = cosine_warmup(optimizer, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=C.USE_AMP and device.type == "cuda")

    hist = History()
    best_acc, best_state, best_epoch, since_improve = -1.0, None, -1, 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=C.USE_AMP and device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        tr_loss, tr_acc = loss_sum / total, correct / total
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        hist.train_loss.append(tr_loss); hist.train_acc.append(tr_acc)
        hist.val_loss.append(va_loss); hist.val_acc.append(va_acc)

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1

        if log_every and epoch % log_every == 0:
            dt = time.time() - t0
            print(f"  época {epoch+1:02d}/{epochs} | "
                  f"train_loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val_loss {va_loss:.3f} acc {va_acc:.3f} | "
                  f"{dt:.1f}s"
                  + (" *" if best_epoch == epoch else ""))

        if patience and since_improve >= patience:
            print(f"  early stopping (sem melhora há {patience} épocas)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  melhor val_acc {best_acc:.3f} (época {best_epoch+1})")
    return model, hist

def gpu_mem_summary(device):
    if device.type != "cuda":
        return {}
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device) / 1e6,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1e6,
    }
