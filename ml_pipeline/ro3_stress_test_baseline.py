import io
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix

CSV_PATH = Path("./clean_chest_xray/clean_patient_split_70_15_15.csv")
MODEL_DIR = Path("./saved_models")
BASELINE_PATH = MODEL_DIR / "baseline_densenet121.pth"

IMG_SIZE = 224
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_perturbation(img: Image.Image, p_type: str, severity: float) -> Image.Image:
    if p_type == "clean":
        return img
    elif p_type == "brightness":
        return ImageEnhance.Brightness(img).enhance(severity)
    elif p_type == "contrast":
        return ImageEnhance.Contrast(img).enhance(severity)
    elif p_type == "gaussian_noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, severity, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    elif p_type == "jpeg_compression":
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    elif p_type == "thoracic_scale":
        w, h = img.size
        new_w, new_h = int(w * severity), int(h * severity)
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if severity < 1.0:
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
            return canvas
        else:
            left, top = (new_w - w) // 2, (new_h - h) // 2
            return resized.crop((left, top, left + w, top + h))
    return img

class PerturbedCXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, p_type: str, severity: float, transform=None):
        self.df = df.reset_index(drop=True)
        self.p_type = p_type
        self.severity = severity
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = int(self.df.loc[idx, "label"])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = apply_perturbation(img, self.p_type, self.severity)
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def build_densenet121() -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "Accuracy": round(float(acc * 100), 2),
        "Sensitivity": round(float(sens * 100), 2),
        "Specificity": round(float(spec * 100), 2),
        "Precision": round(float(prec * 100), 2),
        "F1_Score": round(float(f1 * 100), 2),
        "ROC_AUC": round(float(auc * 100), 2)
    }

def main():
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["new_split"] == "test"]

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = build_densenet121().to(device)
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    model.eval()

    scenarios = [
        ("Clean (Baseline)", "clean", 1.0),
        ("Low Brightness (0.5x)", "brightness", 0.5),
        ("High Brightness (1.5x)", "brightness", 1.5),
        ("Low Contrast / Haze (0.4x)", "contrast", 0.4),
        ("Gaussian Noise (sigma=25)", "gaussian_noise", 25.0),
        ("Heavy Noise (sigma=45)", "gaussian_noise", 45.0),
        ("JPEG Compression (Q=10)", "jpeg_compression", 10.0),
        ("Thoracic Scale (-18% Vol)", "thoracic_scale", 0.82),
        ("Thoracic Scale (+18% Vol)", "thoracic_scale", 1.18),
    ]

    results = []
    np.random.seed(42)

    for name, p_type, sev in scenarios:
        ds = PerturbedCXRDataset(test_df, p_type, sev, transform=eval_tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=name, leave=False):
                images = images.to(device)
                probs = torch.sigmoid(model(images))
                all_labels.extend(labels.numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

        m = compute_metrics(np.array(all_labels), np.array(all_probs))
        m["Scenario"] = name
        m["Perturbation"] = p_type
        m["Severity"] = sev
        results.append(m)

    pd.DataFrame(results).to_csv(MODEL_DIR / "ro3_baseline_stress_test.csv", index=False)

if __name__ == "__main__":
    main()
