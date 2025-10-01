import yaml, torch
from torch.utils.data import DataLoader
from .train_utils import evaluate
from ..data.transforms import build_transforms
from ..data.dataset import CsvImageDataset
from ..models.factory import build_model

def run(cfg_path, weights, split="test"):
    with open(cfg_path, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    csv = cfg["data"][f"{split}_csv"]
    t_eval = build_transforms(cfg["data"]["img_size"], cfg["data"]["mean"], cfg["data"]["std"], False, None)
    ds = CsvImageDataset(csv, t_eval)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"])
    model = build_model(backbone=cfg["model"]["backbone"],
                        num_classes=cfg["data"]["num_classes"],
                        pretrained=False,
                        head_dropout=cfg["model"]["head_dropout"],
                        attention=cfg["model"]["attention"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    m, cm = evaluate(model, dl, device, cfg["data"]["num_classes"])
    print("Metrics:", m)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--weights", default="outputs/best.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run(args.cfg, args.weights, args.split)
