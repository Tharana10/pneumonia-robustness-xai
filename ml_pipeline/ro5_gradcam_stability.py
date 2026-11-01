import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

CSV_PATH = Path("./clean_chest_xray/clean_patient_split_70_15_15.csv")
MODEL_DIR = Path("./saved_models")
SAMPLE_DIR = MODEL_DIR / "gradcam_samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = MODEL_DIR / "baseline_densenet121.pth"
ROBUST_PATH = MODEL_DIR / "robust_densenet121.pth"

IMG_SIZE = 224
NUM_EVAL_IMAGES = 120
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_densenet121() -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model

class DenseNetGradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        with torch.no_grad():
            raw_features = self.model.features(input_tensor)

        feature_var = raw_features.detach().requires_grad_(True)
        activations = F.relu(feature_var, inplace=False)
        pooled = F.adaptive_avg_pool2d(activations, (1, 1))
        flattened = torch.flatten(pooled, 1)
        logits = self.model.classifier(flattened)

        logits[:, 0].backward()
        gradients = feature_var.grad

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations.detach(), dim=1, keepdim=True)
        cam = F.relu(cam, inplace=False)

        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np

def compute_ssim(map1: np.ndarray, map2: np.ndarray) -> float:
    t1 = torch.tensor(map1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t2 = torch.tensor(map2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    mu1 = F.avg_pool2d(t1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(t2, kernel_size=11, stride=1, padding=5)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(t1 * t1, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(t2 * t2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(t1 * t2, kernel_size=11, stride=1, padding=5) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return float(torch.clamp(ssim_map.mean(), 0.0, 1.0).item())

def compute_pcc(map1: np.ndarray, map2: np.ndarray) -> float:
    flat1, flat2 = map1.flatten(), map2.flatten()
    if np.std(flat1) < 1e-8 or np.std(flat2) < 1e-8:
        return 0.0
    corr = np.corrcoef(flat1, flat2)[0, 1]
    return float(np.clip(corr, -1.0, 1.0))

def compute_lung_vs_border_ratio(heatmap: np.ndarray) -> float:
    total_energy = float(np.sum(heatmap)) + 1e-8
    margin = int(IMG_SIZE * 0.125)
    lung_zone_energy = float(np.sum(heatmap[margin:IMG_SIZE - margin, margin:IMG_SIZE - margin]))
    return (lung_zone_energy / total_energy) * 100.0

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

def apply_jet_colormap(cam: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

def save_visual_overlay(orig_img: Image.Image, cam_base: np.ndarray, cam_rob: np.ndarray, out_path: Path):
    base_rgb = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    hm_base = apply_jet_colormap(cam_base).astype(np.float32)
    hm_rob = apply_jet_colormap(cam_rob).astype(np.float32)

    ov_base = np.clip(0.55 * base_rgb + 0.45 * hm_base, 0, 255).astype(np.uint8)
    ov_rob = np.clip(0.55 * base_rgb + 0.45 * hm_rob, 0, 255).astype(np.uint8)

    combined = np.hstack([base_rgb.astype(np.uint8), ov_base, ov_rob])
    Image.fromarray(combined).save(out_path, quality=95)

def main():
    df = pd.read_csv(CSV_PATH)
    test_df = df[df["new_split"] == "test"]

    norm_sample = test_df[test_df["label"] == 0].sample(n=NUM_EVAL_IMAGES // 2, random_state=SEED)
    pneu_sample = test_df[test_df["label"] == 1].sample(n=NUM_EVAL_IMAGES // 2, random_state=SEED)
    eval_df = pd.concat([norm_sample, pneu_sample]).reset_index(drop=True)

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    baseline_model = build_densenet121().to(device)
    baseline_model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    gradcam_base = DenseNetGradCAM(baseline_model)

    robust_model = build_densenet121().to(device)
    robust_model.load_state_dict(torch.load(ROBUST_PATH, map_location=device))
    gradcam_rob = DenseNetGradCAM(robust_model)

    perturbations = [
        ("Low Brightness (0.5x)", "brightness", 0.5),
        ("High Brightness (1.5x)", "brightness", 1.5),
        ("Low Contrast / Haze (0.4x)", "contrast", 0.4),
        ("Gaussian Noise (sigma=25)", "gaussian_noise", 25.0),
        ("Heavy Noise (sigma=45)", "gaussian_noise", 45.0),
        ("JPEG Compression (Q=10)", "jpeg_compression", 10.0),
        ("Thoracic Scale (-18% Vol)", "thoracic_scale", 0.82),
        ("Thoracic Scale (+18% Vol)", "thoracic_scale", 1.18),
    ]

    clean_maps_base, clean_maps_rob = [], []
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), leave=False):
        with Image.open(row["image_path"]) as img:
            img_rgb = img.convert("RGB")
            tensor = eval_tf(img_rgb).unsqueeze(0).to(device)
            clean_maps_base.append(gradcam_base.generate(tensor))
            clean_maps_rob.append(gradcam_rob.generate(tensor))

    summary_rows = []
    for p_name, p_type, sev in perturbations:
        ssim_b_list, ssim_r_list = [], []
        pcc_b_list, pcc_r_list = [], []
        lung_b_list, lung_r_list = [], []

        np.random.seed(SEED)
        for idx, row in eval_df.iterrows():
            with Image.open(row["image_path"]) as img:
                img_rgb = img.convert("RGB")
                pert_img = apply_perturbation(img_rgb, p_type, sev)
                tensor = eval_tf(pert_img).unsqueeze(0).to(device)

                pb = gradcam_base.generate(tensor)
                pr = gradcam_rob.generate(tensor)

                ssim_b_list.append(compute_ssim(clean_maps_base[idx], pb))
                ssim_r_list.append(compute_ssim(clean_maps_rob[idx], pr))
                pcc_b_list.append(compute_pcc(clean_maps_base[idx], pb))
                pcc_r_list.append(compute_pcc(clean_maps_rob[idx], pr))
                lung_b_list.append(compute_lung_vs_border_ratio(pb))
                lung_r_list.append(compute_lung_vs_border_ratio(pr))

                if idx in [0, 60] and p_type in ["gaussian_noise", "brightness", "thoracic_scale"]:
                    out_file = SAMPLE_DIR / f"sample_{idx}_{p_type}_{sev}.jpg"
                    save_visual_overlay(pert_img, pb, pr, out_file)

        summary_rows.append({
            "Perturbation": p_name,
            "Baseline_SSIM": round(float(np.mean(ssim_b_list)), 4),
            "Robust_SSIM": round(float(np.mean(ssim_r_list)), 4),
            "Baseline_PCC": round(float(np.mean(pcc_b_list)), 4),
            "Robust_PCC": round(float(np.mean(pcc_r_list)), 4),
            "Baseline_Lung_Focus_Pct": round(float(np.mean(lung_b_list)), 2),
            "Robust_Lung_Focus_Pct": round(float(np.mean(lung_r_list)), 2),
        })

    pd.DataFrame(summary_rows).to_csv(MODEL_DIR / "ro5_gradcam_stability_comparison.csv", index=False)

if __name__ == "__main__":
    main()
