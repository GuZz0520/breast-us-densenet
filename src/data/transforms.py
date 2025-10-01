import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(img_size, mean, std, is_train=True, aug_cfg=None):
    aug_cfg = aug_cfg or {}
    if is_train:
        tfs = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5) if aug_cfg.get("flip", True) else A.NoOp(),
            A.Rotate(limit=aug_cfg.get("rotate_limit", 15), p=0.5),
            A.RandomBrightnessContrast(aug_cfg.get("brightness_contrast", 0.2), aug_cfg.get("brightness_contrast", 0.2), p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        tfs = [A.Resize(img_size, img_size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose([t for t in tfs if not isinstance(t, A.NoOp)])
