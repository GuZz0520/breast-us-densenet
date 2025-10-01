import torch
import torch.nn.functional as F
import numpy as np
import cv2

class SimpleGradCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.activations = None
        self.gradients = None
        def fwd_hook(module, inp, out): self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out): self.gradients = grad_out[0].detach()
        self.fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = target_layer.register_backward_hook(bwd_hook)
    def __del__(self):
        try: self.fwd_handle.remove(); self.bwd_handle.remove()
        except Exception: pass
    def __call__(self, input_tensor, class_idx=None):
        device = torch.device("cuda" if self.use_cuda else "cpu")
        input_tensor = input_tensor.to(device)
        self.model.to(device); self.model.zero_grad()
        logits = self.model(input_tensor)
        cls = logits.argmax(dim=1) if class_idx is None else torch.tensor([class_idx]*logits.size(0), device=device)
        loss = logits.gather(1, cls.view(-1,1)).sum()
        loss.backward()
        acts = self.activations; grads = self.gradients
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        cams = []
        N, _, H, W = input_tensor.shape
        for i in range(cam.size(0)):
            m = cam[i]; m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            m = m.detach().cpu().numpy()
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            cams.append(m)
        return np.stack(cams, axis=0)

def overlay_cam_on_image(rgb_np, cam_gray, alpha=0.5):
    import numpy as np, cv2
    if rgb_np.dtype != np.uint8:
        img = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    else:
        img = rgb_np
    h, w = img.shape[:2]
    cam = cam_gray
    if cam.shape != (h, w):
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img).clip(0, 255).astype(np.uint8)
    return overlay
