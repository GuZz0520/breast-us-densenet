import os, csv, re, random, argparse, sys, glob

VALID_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
CLASSES   = ["normal","benign","malignant"]

def is_image(p):
    return os.path.splitext(p)[1].lower() in VALID_EXT

def guess_pid(name):
    base = os.path.basename(name)
    m = re.match(r"([A-Za-z0-9]+)[-_]", base)
    return m.group(1) if m else os.path.splitext(base)[0]

def collect_images(root):
    stats = {}
    rows = []
    for c in CLASSES:
        class_dir = os.path.join(root, c)
        if not os.path.isdir(class_dir):
            print(f"[WARN] class folder not found: {class_dir}", file=sys.stderr)
            stats[c] = 0
            continue
        # 递归所有文件
        paths = [p for p in glob.glob(os.path.join(class_dir, "**", "*"), recursive=True) if os.path.isfile(p)]
        imgs = [p for p in paths if is_image(p)]
        stats[c] = len(imgs)
        for p in imgs:
            rows.append([p, c, guess_pid(p)])
    return rows, stats

def write_split(rows, outdir, val_ratio=0.15, test_ratio=0.15, seed=2025):
    # 病人级划分
    pids = sorted(set(r[2] for r in rows))
    random.seed(seed); random.shuffle(pids)
    n = len(pids); n_test = int(n*test_ratio); n_val = int(n*val_ratio)
    test_pids = set(pids[:n_test])
    val_pids  = set(pids[n_test:n_test+n_val])
    train_pids= set(pids[n_test+n_val:])

    os.makedirs(outdir, exist_ok=True)
    for split, pidset in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
        out_csv = os.path.join(outdir, f"{split}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["path","label","patient_id"])
            nrows = 0
            for r in rows:
                if r[2] in pidset:
                    w.writerow(r); nrows += 1
        print(f"[OK] wrote {out_csv} with {nrows} rows")

def main(root, outdir, val_ratio=0.15, test_ratio=0.15):
    print(f"[INFO] root={os.path.abspath(root)}  outdir={os.path.abspath(outdir)}")
    print(f"[INFO] expecting class subfolders: {', '.join(CLASSES)}")
    rows, stats = collect_images(root)
    total = sum(stats.values())
    print(f"[INFO] found images: {stats}  total={total}")
    if total == 0:
        print("[ERROR] No images found. Check folder names, file extensions, or nesting.", file=sys.stderr)
        sys.exit(1)
    write_split(rows, outdir, val_ratio, test_ratio)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root containing class subfolders")
    ap.add_argument("--outdir", required=True, help="output directory for CSVs")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()
    main(args.root, args.outdir, )
