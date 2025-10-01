# scripts/check_masks_in_csvs.py
import os, re, pandas as pd

def has_mask(p: str) -> bool:
    p = p.lower()
    return ("mask" in os.path.basename(p)) or re.search(r"(\\|/)(mask|masks)(\\|/)", p) is not None

for split in ["train","val","test"]:
    csv_path = f"data/{split}.csv"
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found"); continue
    df = pd.read_csv(csv_path)
    n_all = len(df)
    n_mask = sum(df["path"].astype(str).map(has_mask))
    print(f"{csv_path}: total={n_all}, mask_like={n_mask}")
    if n_mask:
        print("  -> examples:")
        print(df[df["path"].astype(str).map(has_mask)].head(5))
