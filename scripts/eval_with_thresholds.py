# scripts/eval_with_thresholds.py
import os, sys, json, argparse, yaml, numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.data.transforms import build_transforms
from src.data.dataset import CsvImageDataset
from src.models.factory import build_model
torch.set_grad_enabled(False)

def load_thresholds(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    thr = np.array(obj["thresholds"], dtype=np.float32)
    return thr, obj

def load_everything(cfg_path, weights, split):
    with open(cfg_path, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    t_eval = build_transforms(cfg["data"]["img_size"], cfg["data"]["mean"], cfg["data"]["std"], False, None)
    ds = CsvImageDataset(cfg["data"][f"{split}_csv"], t_eval)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)
    model = build_model(backbone=cfg["model"]["backbone"],
                        num_classes=cfg["data"]["num_classes"],
                        pretrained=False,
                        head_dropout=cfg["model"]["head_dropout"],
                        attention=cfg["model"]["attention"]).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state); model.eval()
    return cfg, device, model, dl

def predict_probs(model, device, dl):
    probs, labels = [], []
    for xb, yb in tqdm(dl, desc="predict"):
        xb = xb.to(device)
        p = torch.softmax(model(xb), dim=1).cpu().numpy()
        probs.append(p); labels.append(yb.numpy())
    return np.concatenate(probs, 0), np.concatenate(labels, 0)

def argmax_with_thresholds(probs, thr):
    adj = probs / thr[None, :]
    return adj.argmax(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--weights", default="outputs/best.pt")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--thresholds", default="outputs/thresholds.json")
    args = ap.parse_args()

    thr, meta = load_thresholds(args.thresholds)
    cfg, device, model, dl = load_everything(args.cfg, args.weights, args.split)
    probs, labels = predict_probs(model, device, dl)
    yhat = argmax_with_thresholds(probs, thr)

    acc = accuracy_score(labels, yhat)
    macro_f1 = f1_score(labels, yhat, average="macro")
    # One-vs-Rest macro AUC（需要每类至少有正样本）
    try:
        macro_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = float("nan")
    cm = confusion_matrix(labels, yhat)

    print("\n=== Evaluation on {} with thresholds ===".format(args.split))
    print("thresholds :", thr.tolist())
    print(f"Accuracy   : {acc:.6f}")
    print(f"Macro-F1   : {macro_f1:.6f}")
    print(f"Macro-AUC  : {macro_auc:.6f}")
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
