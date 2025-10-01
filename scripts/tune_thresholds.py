# scripts/tune_thresholds.py
import os, sys, json, argparse, yaml, itertools, numpy as np
from tqdm import tqdm

# 确保能 import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from src.data.transforms import build_transforms
from src.data.dataset import CsvImageDataset
from src.models.factory import build_model
from src.engine.train_utils import evaluate  # 复用评估指标
torch.set_grad_enabled(False)

CLASSES = ["normal","benign","malignant"]

def load_model_and_data(cfg_path, weights, split):
    with open(cfg_path, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # dataset / dataloader
    t_eval = build_transforms(cfg["data"]["img_size"], cfg["data"]["mean"], cfg["data"]["std"], False, None)
    csv_path = cfg["data"][f"{split}_csv"]
    ds = CsvImageDataset(csv_path, t_eval)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # model
    model = build_model(backbone=cfg["model"]["backbone"],
                        num_classes=cfg["data"]["num_classes"],
                        pretrained=False,
                        head_dropout=cfg["model"]["head_dropout"],
                        attention=cfg["model"]["attention"]).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state); model.eval()
    return cfg, device, model, dl

def collect_probs_labels(model, device, dl):
    probs, labels = [], []
    for xb, yb in tqdm(dl, desc="collect"):
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p); labels.append(yb.numpy())
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return probs, labels

def predict_with_thresholds(probs, thresholds):
    # thresholds: list/array of length C
    adj = probs / thresholds[None, :]  # shape [N, C]
    return adj.argmax(axis=1)

def grid_search_thresholds(probs, labels, grid):
    """
    grid: list of candidate thresholds for each class, e.g., [0.3,0.32,...,0.9]
    返回最佳 thresholds（按 accuracy 最大），以及对应的指标
    """
    C = probs.shape[1]
    best = {"acc": -1.0, "thr": None, "macro_f1": None}
    from sklearn.metrics import f1_score, accuracy_score
    for thr_tuple in itertools.product(grid, repeat=C):
        thr = np.array(thr_tuple, dtype=np.float32)
        yhat = predict_with_thresholds(probs, thr)
        acc = accuracy_score(labels, yhat)
        macro_f1 = f1_score(labels, yhat, average="macro")
        if acc > best["acc"]:
            best = {"acc": acc, "thr": thr.tolist(), "macro_f1": macro_f1}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--weights", default="outputs/best.pt")
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--grid", default="0.30:0.90:0.02", help="start:stop:step for thresholds")
    ap.add_argument("--out", default="outputs/thresholds.json")
    args = ap.parse_args()

    # 构造阈值网格
    s, e, st = map(float, args.grid.split(":"))
    grid = list(np.arange(s, e+1e-6, st))

    cfg, device, model, dl = load_model_and_data(args.cfg, args.weights, args.split)
    probs, labels = collect_probs_labels(model, device, dl)
    best = grid_search_thresholds(probs, labels, grid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "classes": CLASSES,
            "thresholds": best["thr"],
            "val_split": args.split,
            "grid": args.grid,
            "val_acc": best["acc"],
            "val_macro_f1": best["macro_f1"],
        }, f, indent=2, ensure_ascii=False)

    print("\n=== Best on {} ===".format(args.split))
    print("thresholds:", best["thr"])
    print("val Acc   :", round(best["acc"], 6))
    print("val Macro-F1:", round(best["macro_f1"], 6))
    print("saved to  :", args.out)

if __name__ == "__main__":
    main()
