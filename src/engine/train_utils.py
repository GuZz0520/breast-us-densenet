import os, yaml, torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from ..utils.metrics import compute_metrics

def save_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    y_true, y_pred = [], []
    y_prob_all = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), torch.as_tensor(y, device=device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(prob.argmax(1).cpu().tolist())
        y_prob_all.append(prob.cpu())
    y_prob = torch.cat(y_prob_all, dim=0).numpy()
    m = compute_metrics(y_true, y_pred, y_prob, num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return m, cm

def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0, scaler=None):
    import numpy as np
    import torch
    from torch.amp import autocast  # 新API
    from tqdm import tqdm

    model.train()
    total = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), torch.as_tensor(y, device=device)

        # mixup（可选）
        if mixup_alpha and mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(x.size(0), device=device)
            x = lam * x + (1 - lam) * x[idx]
            y_a, y_b = y, y[idx]
        else:
            y_a = y_b = None  # 安全占位

        optimizer.zero_grad(set_to_none=True)

        # 混合精度（仅在 CUDA 时启用）
        use_amp = (scaler is not None)
        ctx = autocast(device_type="cuda", enabled=use_amp) if device.type == "cuda" else \
              autocast(device_type="cpu", enabled=False)

        with ctx:
            logits = model(x)
            if mixup_alpha and mixup_alpha > 0:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.detach().item())

    return total / max(1, len(loader))