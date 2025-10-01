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

## 1. Features

- 🔁 **Reproducible**: fixed seeds, config saved to `outputs/config_used.yaml`
- ⚖️ **Imbalance-aware**: Focal / Class-Balanced CE (+ label smoothing, optional mixup)
- 🧊 **Staged freezing**: head → block4 → full
- 🔍 **Grad-CAM**: pure PyTorch; auto-selects DenseNet block4 target conv
- 🖥️ **Gradio** app; **ONNX** export; simple CLI inference

---

## 2. Environment

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
