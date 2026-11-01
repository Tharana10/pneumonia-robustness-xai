import json
from pathlib import Path
from PIL import Image
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
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CXRDataset(Dataset):
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
            if self.transform:
                img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def build_densenet121() -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model

def set_parameter_requires_grad(model: nn.Module, unfreeze_last_block: bool = False):
    for param in model.features.parameters():
        param.requires_grad = False
    if unfreeze_last_block:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True

def compute_clinical_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    return {
        "Accuracy": round(float(acc * 100), 2),
        "Sensitivity": round(float(sensitivity * 100), 2),
        "Specificity": round(float(specificity * 100), 2),
        "Precision": round(float(precision * 100), 2),
        "F1_Score": round(float(f1 * 100), 2),
        "ROC_AUC": round(float(roc_auc * 100), 2),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_clinical_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics

def main():
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["new_split"] == "train"]
    val_df = df[df["new_split"] == "val"]
    test_df = df[df["new_split"] == "test"]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(CXRDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(CXRDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(CXRDataset(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_neg = (train_df["label"] == 0).sum()
    num_pos = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    model = build_densenet121().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    best_model_path = MODEL_DIR / "baseline_densenet121.pth"

    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            set_parameter_requires_grad(model, unfreeze_last_block=False)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        elif epoch == 4:
            set_parameter_requires_grad(model, unfreeze_last_block=True)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

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

        train_loss /= len(train_loader.dataset)
        val_loss, val_metrics = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_metrics['F1_Score']:.2f}%")

        if val_metrics["F1_Score"] > best_val_f1:
            best_val_f1 = val_metrics["F1_Score"]
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    _, test_metrics = evaluate(model, test_loader, criterion)
    with open(MODEL_DIR / "ro2_baseline_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    main()
