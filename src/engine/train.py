import os
import yaml
import torch
from torch import amp
from torch.utils.data import DataLoader

from ..utils.seed import set_seed
from ..data.transforms import build_transforms
from ..data.dataset import CsvImageDataset
from ..models.factory import build_model
from ..losses.focal import FocalLoss
from ..losses.class_balanced import class_balanced_weight
from .train_utils import evaluate, train_one_epoch, save_yaml


def build_criterion(cfg, device, train_csv):
    # focal
    if cfg["train"]["loss"] == "focal":
        return FocalLoss(
            gamma=cfg["train"]["focal_gamma"],
            label_smoothing=cfg["train"]["label_smoothing"],
        ).to(device)

    # class-balanced cross-entropy
    if cfg["train"]["loss"] == "cbce":
        import pandas as pd
        from ..data.dataset import label_to_int

        df = pd.read_csv(train_csv)
        counts = df["label"].value_counts()
        order = sorted(counts.index, key=lambda x: label_to_int(x))
        cls_counts = [counts.get(k, 0) for k in order]
        w = class_balanced_weight(cls_counts, beta=0.9999, device=device)
        return torch.nn.CrossEntropyLoss(
            weight=w, label_smoothing=cfg["train"]["label_smoothing"]
        ).to(device)

    # plain CE
    return torch.nn.CrossEntropyLoss(
        label_smoothing=cfg["train"]["label_smoothing"]
    ).to(device)


def run(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # data & loaders
    img_size = cfg["data"]["img_size"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    t_train = build_transforms(img_size, mean, std, True, cfg["data"]["aug"])
    t_eval = build_transforms(img_size, mean, std, False, None)

    ds_train = CsvImageDataset(cfg["data"]["train_csv"], t_train)
    ds_val = CsvImageDataset(cfg["data"]["val_csv"], t_eval)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    # model
    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        head_dropout=cfg["model"]["head_dropout"],
        attention=cfg["model"]["attention"],
    ).to(device)

    # freeze schedule helper
    freeze_schedule = cfg["model"].get("freeze_schedule", [])

    def set_backbone_trainable(part: str):
        # all frozen by default
        for p in model.features.parameters():
            p.requires_grad = False
        # unfreeze last block
        if part in ("block4", "full"):
            for name, p in model.features.named_parameters():
                if (
                    "denseblock4" in name
                    or "norm5" in name
                    or "transition3" in name
                ):
                    p.requires_grad = True
        # unfreeze all
        if part == "full":
            for p in model.features.parameters():
                p.requires_grad = True

    # opt / sched / loss / scaler
    criterion = build_criterion(cfg, device, cfg["data"]["train_csv"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"]
        )
        if cfg["train"]["scheduler"] == "cosine"
        else None
    )
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    # bookkeeping
    best_metric_name = cfg["save"]["best_metric"]
    best_val = -1e9
    no_improve = 0
    out_dir = cfg["save"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(out_dir, "config_used.yaml"))

    # training
    for phase in freeze_schedule:
        set_backbone_trainable(phase)
        print(f"[Freeze Schedule] phase: {phase}")
        for epoch in range(cfg["train"]["epochs"]):
            loss = train_one_epoch(
                model,
                dl_train,
                optimizer,
                criterion,
                device,
                mixup_alpha=cfg["train"]["mixup_alpha"],
                scaler=scaler,
            )

            if scheduler:
                scheduler.step()

            metrics, _ = evaluate(
                model, dl_val, device, cfg["data"]["num_classes"]
            )
            cur = (
                metrics["macro_f1"]
                if best_metric_name == "macro_f1"
                else metrics["macro_auc"]
            )

            print(
                f"epoch {epoch+1:03d} | loss {loss:.4f} | "
                f"acc {metrics['acc']:.4f} | macro_f1 {metrics['macro_f1']:.4f} | "
                f"macro_auc {metrics['macro_auc']:.4f}"
            )

            if cur > best_val:
                best_val = cur
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            else:
                no_improve += 1
                if no_improve >= cfg["train"]["early_stopping"]:
                    print("Early stopping at epoch", epoch + 1)
                    break

    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    print("Best", best_metric_name, "=", best_val)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    args = ap.parse_args()
    run(args.cfg)

