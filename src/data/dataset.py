import cv2, pandas as pd
from torch.utils.data import Dataset
import numpy as np

CLASS_MAP = {"normal":0,"benign":1,"malignant":2}

def label_to_int(x):
    if isinstance(x, (int, np.integer)): return int(x)
    return CLASS_MAP[str(x).lower()]

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        assert {"path","label","patient_id"}.issubset(self.df.columns), \
            "CSV must contain path,label,patient_id"
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = label_to_int(row.label)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label
