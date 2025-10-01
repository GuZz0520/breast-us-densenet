# breast-us-densenet

Reproducible PyTorch pipeline for **3-class breast ultrasound** (normal / benign / malignant) based on **DenseNet121**, with:
- configurable training & evaluation
- class-imbalance losses (Focal / Class-Balanced CE)
- staged freezing + AMP
- **built-in Grad-CAM** (pure PyTorch, no external deps)
- Gradio demo, ONNX export

> **Paper:** _add link here_  
> **Repo:** https://github.com/GuZz0520/breast-us-densenet  
> **Model weights (Release):(https://github.com/GuZz0520/breast-us-densenet/releases/tag/v0.1.0)

---

1. Features

- 🔁 **Reproducible**: fixed seeds, config saved to `outputs/config_used.yaml`
- ⚖️ **Imbalance-aware**: Focal / Class-Balanced CE (+ label smoothing, optional mixup)
- 🧊 **Staged freezing**: head → block4 → full
- 🔍 **Grad-CAM**: pure PyTorch; auto-selects DenseNet block4 target conv
- 🖥️ **Gradio** app; **ONNX** export; simple CLI inference

---

2. Environment

> Verified on **Windows + Python 3.13**. (Linux/Mac 也可，命令略有差异)

```bat
:: create env
conda create -n BreastUS python=3.13 -y
conda activate BreastUS

:: install deps (no torch inside)
python -m pip install --upgrade pip
python -m pip install -r requirements-no-torch.txt

:: install PyTorch (choose one)
:: CPU:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
:: OR CUDA (replace cuXXX with your CUDA build):
:: python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
3. Data preparation
项目使用 CSV 作为清单（不直接读整文件夹）。CSV 必含两列：
path,label
D:/.../img_001.png,benign
D:/.../img_002.png,malignant
...
三个文件放在 data/ 下：
data/train.csv
data/val.csv
data/test.csv
3.1 从文件夹生成 CSV（可选）
你的原始数据若按 root/<class>/*.png 组织，可用脚本自动生成：
python scripts\make_csv_from_folders.py --root data\images --outdir data
3.2 过滤掩码（mask）文件
为避免把分割标注图当作分类样本，先清理 CSV：python -u scripts\filter_out_masks.py
4. Train / Evaluate
4.1 训练
python -m src.engine.train --cfg configs\baseline.yaml
4.2 评估（val / test）
python -m src.engine.evaluate --cfg configs\baseline.yaml --weights outputs\best.pt --split val
python -m src.engine.evaluate --cfg configs\baseline.yaml --weights outputs\best.pt --split test
5. Inference & Visualization
5.1 CLI 预测 + Grad-CAM 可视化
python -m src.infer --cfg configs\baseline.yaml --weights outputs\best.pt ^
  --image D:\path\to\one_image.png --out outputs\cam_example.jpg
5.2 Gradio Web App
python -u scripts\app_gradio.py
:: 启动后控制台显示本地地址，如 http://127.0.0.1:7861 或 7862
6. Results (clean test set)
训练数据清理了 mask 后的 正式结果（仅在 test 上评一次）：
Accuracy: 0.7983
Macro-F1: 0.7733
Macro-AUC: 0.9024

为了避免把大文件提交到 Git，模型权重请从 Releases 下载：
best.pt（main weights）
config_used.yaml（训练时真实配置）
model.onnx（可选）
