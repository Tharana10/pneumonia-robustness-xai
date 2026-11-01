import re
from pathlib import Path
from PIL import Image
import imagehash
import kagglehub
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

CLEAN_OUTPUT_DIR = Path("./clean_chest_xray")
RANDOM_SEED = 42

def get_dataset_root() -> Path:
    base_path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    if (base_path / "chest_xray" / "train").exists():
        return base_path / "chest_xray"
    elif (base_path / "train").exists():
        return base_path
    raise FileNotFoundError(f"Dataset folders not found in {base_path}")

def extract_patient_id(filename: str, label: str) -> str:
    name = Path(filename).stem
    person_match = re.search(r"(person\d+)", name, re.IGNORECASE)
    if person_match:
        return f"P_{person_match.group(1).lower()}"
    im_match = re.search(r"(IM-\d+)", name, re.IGNORECASE)
    if im_match:
        return f"N_{im_match.group(1).upper()}"
    return f"{label}_{name}"

def scan_and_deduplicate(raw_dir: Path):
    records = []
    seen_hashes = {}
    duplicates = []

    splits = ["train", "val", "test"]
    classes = {"NORMAL": 0, "PNEUMONIA": 1}

    all_files = []
    for s in splits:
        for cls_name in classes.keys():
            folder = raw_dir / s / cls_name
            if folder.exists():
                for img_path in folder.glob("*.*"):
                    if img_path.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                        all_files.append((img_path, cls_name, classes[cls_name], s))

    for img_path, cls_name, label_id, orig_split in tqdm(all_files, desc="Scanning images"):
        try:
            with Image.open(img_path) as img:
                phash_val = str(imagehash.phash(img))
                width, height = img.size
        except Exception as e:
            print(f"Skipped {img_path}: {e}")
            continue

        if phash_val in seen_hashes:
            duplicates.append({
                "duplicate_path": str(img_path),
                "original_path": seen_hashes[phash_val],
                "phash": phash_val,
                "label": cls_name,
            })
            continue

        seen_hashes[phash_val] = str(img_path)
        patient_id = extract_patient_id(img_path.name, cls_name)

        records.append({
            "image_path": str(img_path),
            "filename": img_path.name,
            "patient_id": patient_id,
            "label_name": cls_name,
            "label": label_id,
            "original_split": orig_split,
            "phash": phash_val,
            "width": width,
            "height": height,
        })

    df_clean = pd.DataFrame(records)
    df_dups = pd.DataFrame(duplicates)
    print(f"Scanned: {len(all_files)} | Duplicates: {len(df_dups)} | Clean: {len(df_clean)}")
    return df_clean, df_dups

def patient_level_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["label"], df["patient_id"])):
        df.loc[val_idx, "fold"] = fold_idx

    df["new_split"] = df["fold"].apply(lambda f: "train" if f < 14 else ("val" if f < 17 else "test"))

    train_pats = set(df[df["new_split"] == "train"]["patient_id"])
    val_pats = set(df[df["new_split"] == "val"]["patient_id"])
    test_pats = set(df[df["new_split"] == "test"]["patient_id"])

    assert len(train_pats & val_pats) == 0
    assert len(train_pats & test_pats) == 0
    assert len(val_pats & test_pats) == 0
    return df

def save_curation_outputs(df: pd.DataFrame, df_dups: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "clean_patient_split_70_15_15.csv", index=False)
    df_dups.to_csv(output_dir / "removed_phash_duplicates.csv", index=False)

    for split_name in ["train", "val", "test"]:
        sub = df[df["new_split"] == split_name]
        norm_cnt = len(sub[sub["label_name"] == "NORMAL"])
        pneu_cnt = len(sub[sub["label_name"] == "PNEUMONIA"])
        pat_cnt = sub["patient_id"].nunique()
        print(f"{split_name.upper()} -> Total: {len(sub)}, Normal: {norm_cnt}, Pneumonia: {pneu_cnt}, Patients: {pat_cnt}")

if __name__ == "__main__":
    raw_dir = get_dataset_root()
    df_clean, df_dups = scan_and_deduplicate(raw_dir)
    df_split = patient_level_split(df_clean, seed=RANDOM_SEED)
    save_curation_outputs(df_split, df_dups, CLEAN_OUTPUT_DIR)
