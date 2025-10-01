import os, yaml, torch, cv2
import numpy as np
from .data.transforms import build_transforms
from .models.factory import build_model
from .engine.grad_cam import SimpleGradCAM, overlay_cam_on_image

def run(cfg_path, weights, image_path, out_path="outputs/cam.jpg"):
    with open(cfg_path, encoding="utf-8") as f: cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    t_eval = build_transforms(cfg["data"]["img_size"], cfg["data"]["mean"], cfg["data"]["std"], False, None)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR); assert img_bgr is not None, image_path
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = t_eval(image=img_rgb)["image"].unsqueeze(0).to(device)

    model = build_model(backbone=cfg["model"]["backbone"],
                        num_classes=cfg["data"]["num_classes"],
                        pretrained=False,
                        head_dropout=cfg["model"]["head_dropout"],
                        attention=cfg["model"]["attention"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(prob.argmax())

    target_layer = model.features.denseblock4[-1].conv2
    cam = SimpleGradCAM(model=model, target_layer=target_layer, use_cuda=(device.type=="cuda"))
    grayscale_cam = cam(input_tensor)[0]
    vis = overlay_cam_on_image(img_rgb, grayscale_cam, alpha=0.45)
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("pred:", pred, "prob:", prob, "cam saved to:", out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--weights", default="outputs/best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/cam.jpg")
    args = ap.parse_args()
    run(args.cfg, args.weights, args.image, args.out)
