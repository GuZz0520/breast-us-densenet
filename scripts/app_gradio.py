# scripts/app_gradio.py
import os
import sys
import traceback
import numpy as np
from PIL import Image
import gradio as gr

# --- 确保项目根目录在 sys.path（直接运行 /scripts 下的文件时必需） ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------------------

print("[app] starting...")

CFG_PATH = "configs/baseline.yaml"
WEIGHTS = "outputs/best.pt"
CLASSES = ["normal", "benign", "malignant"]

# 懒加载运行时对象
_runtime = {
    "ready": False,
    "err": None,
    "cfg": None,
    "device": None,
    "t_eval": None,
    "model": None,
    "cam": None,
}


def _pick_gradcam_target_layer(model):
    """
    稳健地为 torchvision DenseNet121 选择 Grad-CAM 的目标层：
    取 features.denseblock4 的最后一个 DenseLayer 的 conv2
    """
    # denseblock4 是 _DenseBlock，children() 返回 denselayer1..N
    db4_children = list(model.features.denseblock4.children())
    if not db4_children:
        raise RuntimeError("denseblock4 has no children; unexpected model structure")
    last_layer = db4_children[-1]

    # 优先 conv2 -> 再尝试 conv1 -> 再在该层内找任意 Conv2d
    import torch.nn as nn
    target = getattr(last_layer, "conv2", None)
    if target is None:
        target = getattr(last_layer, "conv1", None)
    if target is None:
        for m in last_layer.modules():
            if isinstance(m, nn.Conv2d):
                target = m
                break
    if target is None:
        raise RuntimeError("Cannot locate a Conv2d in last denselayer for Grad-CAM")
    return target


def lazy_init():
    """首次推理时再加载重量级模块/模型，避免 launch 前卡住"""
    if _runtime["ready"] or _runtime["err"] is not None:
        return

    try:
        print("[app] lazy_init: importing modules...")
        import yaml
        import torch
        import cv2  # noqa: F401
        from src.data.transforms import build_transforms
        from src.models.factory import build_model
        from src.engine.grad_cam import SimpleGradCAM, overlay_cam_on_image  # noqa: F401

        print("[app] lazy_init: loading cfg:", CFG_PATH)
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        t_eval = build_transforms(
            cfg["data"]["img_size"], cfg["data"]["mean"], cfg["data"]["std"], False, None
        )

        print("[app] lazy_init: building model...")
        model = build_model(
            backbone=cfg["model"]["backbone"],
            num_classes=cfg["data"]["num_classes"],
            pretrained=False,
            head_dropout=cfg["model"]["head_dropout"],
            attention=cfg["model"]["attention"],
        ).to(device)

        print("[app] lazy_init: loading weights:", WEIGHTS)
        state = torch.load(WEIGHTS, map_location=device)
        model.load_state_dict(state)
        model.eval()

        print("[app] lazy_init: creating Grad-CAM handle...")
        target_layer = _pick_gradcam_target_layer(model)
        from src.engine.grad_cam import SimpleGradCAM  # 重复导入只为清晰
        cam = SimpleGradCAM(model=model, target_layer=target_layer, use_cuda=(device.type == "cuda"))

        _runtime.update(
            dict(ready=True, cfg=cfg, device=device, t_eval=t_eval, model=model, cam=cam)
        )
        print("[app] lazy_init: ready ✓")

    except Exception as e:
        _runtime["err"] = f"{type(e).__name__}: {e}"
        print("[app] lazy_init: ERROR\n" + "".join(traceback.format_exc()), file=sys.stderr)


def infer_fn(pil_img: Image.Image):
    """Gradio 回调：图片 -> (类别, 概率, CAM图)"""
    lazy_init()
    if _runtime["err"] is not None:
        return f"初始化失败：{_runtime['err']}", {}, None
    if not _runtime["ready"]:
        return "正在初始化，请稍后重试", {}, None

    import torch
    from src.engine.grad_cam import overlay_cam_on_image

    img_rgb = np.array(pil_img.convert("RGB"))
    x = _runtime["t_eval"](image=img_rgb)["image"].unsqueeze(0).to(_runtime["device"])

    with torch.no_grad():
        logits = _runtime["model"](x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
        pred = int(np.argmax(prob))

    gray = _runtime["cam"](x)[0]  # [0,1]
    vis = overlay_cam_on_image(img_rgb, gray, alpha=0.45)
    vis_pil = Image.fromarray(vis)

    prob_dict = {CLASSES[i]: float(p) for i, p in enumerate(prob)}
    return CLASSES[pred], prob_dict, vis_pil


# Gradio 界面
demo = gr.Interface(
    fn=infer_fn,
    inputs=gr.Image(type="pil", label="上传乳腺超声图像"),
    outputs=[
        gr.Label(label="预测类别"),
        gr.JSON(label="各类概率"),
        gr.Image(type="pil", label="Grad-CAM 可视化"),
    ],
    title="Breast US Classification (DenseNet121)",
    description="首次推理时将初始化并加载模型，请稍等片刻。",
    flagging_mode="never",  # 代替旧的 allow_flagging
)

if __name__ == "__main__":
    print("[app] launching gradio...")
    # 优先读环境变量端口；否则自动尝试 7861–7866
    env_port = os.getenv("GRADIO_SERVER_PORT")
    ports = [int(env_port)] if env_port else [7861, 7862, 7863, 7864, 7865, 7866]
    for port in ports:
        try:
            print(f"[app] trying port {port} ...")
            demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                debug=True,
                show_error=True,
            )
            break
        except OSError:
            print(f"[app] port {port} busy, trying next...")
    else:
        raise RuntimeError(f"No free port found in {ports}")

