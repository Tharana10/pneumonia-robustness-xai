import io
import random
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
ROBUST_PATH = MODEL_DIR / "robust_densenet121.pth"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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

def apply_quality_scale_augmentation(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = apply_perturbation(img, "thoracic_scale", random.uniform(0.80, 1.20))
    if random.random() < 0.6:
        img = apply_perturbation(img, "brightness", random.uniform(0.45, 1.55))
    if random.random() < 0.5:
        img = apply_perturbation(img, "contrast", random.uniform(0.35, 1.40))
    if random.random() < 0.45:
        img = apply_perturbation(img, "gaussian_noise", random.uniform(10.0, 48.0))
    if random.random() < 0.35:
        img = apply_perturbation(img, "jpeg_compression", random.randint(8, 35))
    return img

class RobustTrainCXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = int(self.df.loc[idx, "label"])
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if random.random() < 0.75:
                img = apply_quality_scale_augmentation(img)
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

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
        "ROC_AUC": round(float(auc * 100), 2),
    }

def main():
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["new_split"] == "train"]
    val_df = df[df["new_split"] == "val"]
    test_df = df[df["new_split"] == "test"]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(RobustTrainCXRDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(RobustTrainCXRDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_densenet121().to(device)
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))

    for param in model.features.parameters():
        param.requires_grad = False
    for block in [model.features.denseblock3, model.features.transition3, model.features.denseblock4, model.features.norm5]:
        for param in block.parameters():
            param.requires_grad = True

    num_neg = (train_df["label"] == 0).sum()
    num_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_labels, val_probs = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                probs = torch.sigmoid(model(images))
                val_labels.extend(labels.numpy().flatten())
                val_probs.extend(probs.cpu().numpy().flatten())

        val_m = compute_metrics(np.array(val_labels), np.array(val_probs))
        if val_m["F1_Score"] > best_val_f1:
            best_val_f1 = val_m["F1_Score"]
            torch.save(model.state_dict(), ROBUST_PATH)

    model.load_state_dict(torch.load(ROBUST_PATH, map_location=device))
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

    base_df = pd.read_csv(MODEL_DIR / "ro3_baseline_stress_test.csv")
    comparison_rows = []
    np.random.seed(42)

    for idx, (name, p_type, sev) in enumerate(scenarios):
        ds = PerturbedCXRDataset(test_df, p_type, sev, transform=eval_tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                probs = torch.sigmoid(model(images))
                all_labels.extend(labels.numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

        rob_m = compute_metrics(np.array(all_labels), np.array(all_probs))
        base_f1 = base_df.loc[idx, "F1_Score"]
        base_spec = base_df.loc[idx, "Specificity"]

        comparison_rows.append({
            "Scenario": name,
            "Baseline_Acc": base_df.loc[idx, "Accuracy"],
            "Robust_Acc": rob_m["Accuracy"],
            "Baseline_Spec": base_spec,
            "Robust_Spec": rob_m["Specificity"],
            "Spec_Improvement": round(rob_m["Specificity"] - base_spec, 2),
            "Baseline_Sens": base_df.loc[idx, "Sensitivity"],
            "Robust_Sens": rob_m["Sensitivity"],
            "Baseline_F1": base_f1,
            "Robust_F1": rob_m["F1_Score"],
            "F1_Improvement": round(rob_m["F1_Score"] - base_f1, 2),
            "Baseline_AUC": base_df.loc[idx, "ROC_AUC"],
            "Robust_AUC": rob_m["ROC_AUC"],
        })

    pd.DataFrame(comparison_rows).to_csv(MODEL_DIR / "ro4_baseline_vs_robust_comparison.csv", index=False)

if __name__ == "__main__":
    main()
