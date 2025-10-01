import os, re, pandas as pd

def is_mask_path(p):
    p = str(p).lower()
    return ("mask" in os.path.basename(p)) or (re.search(r"(\\|/)(mask|masks)(\\|/)", p) is not None)

for split in ["train","val","test"]:
    csv_path = f"data/{split}.csv"
    if not os.path.exists(csv_path):
        continue
    bak = csv_path + ".bak"
    if not os.path.exists(bak):
        os.replace(csv_path, bak)
        print(f"[backup] {csv_path} -> {bak}")
    df = pd.read_csv(bak)
    before = len(df)
    df2 = df[~df["path"].map(is_mask_path)].copy()
    after = len(df2)
    df2.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[clean]  {csv_path}: {before} -> {after} (removed {before-after} mask rows)")
