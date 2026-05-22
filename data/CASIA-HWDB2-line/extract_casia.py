import pickle
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


PARQUET_FILES = {
    "train": "train.parquet",
    "val": "validation.parquet",
    "test": "test.parquet",
}

OUT_DIR = Path("/home/filippo/projects/Papers/htr_vmamba/data/CASIA-HWDB2-line/")
LINES_DIR = OUT_DIR / "lines"
LINES_DIR.mkdir(parents=True, exist_ok=True)

labels = {}
global_idx = 0


def extract_image(cell):
    """
    Hugging Face parquet image fields are usually stored as:
      {'bytes': ..., 'path': ...}
    or occasionally as raw bytes.
    """
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return Image.open(pd.io.common.BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path") is not None:
            return Image.open(cell["path"]).convert("RGB")

    if isinstance(cell, bytes):
        return Image.open(pd.io.common.BytesIO(cell)).convert("RGB")

    if isinstance(cell, Image.Image):
        return cell.convert("RGB")

    raise TypeError(f"Unsupported image format: {type(cell)}")


for split, parquet_name in PARQUET_FILES.items():
    parquet_path = Path(parquet_name)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing file: {parquet_path}")

    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    if "image" not in df.columns:
        raise KeyError(f"'image' column not found in {parquet_path}. Columns: {df.columns}")

    if "text" not in df.columns:
        raise KeyError(f"'text' column not found in {parquet_path}. Columns: {df.columns}")

    ln_path = OUT_DIR / f"{split}.ln"

    with open(ln_path, "w", encoding="utf-8") as ln_file:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Exporting {split}"):
            img_name = f"img_{global_idx}.png"
            txt_name = f"img_{global_idx}.txt"

            img_path = LINES_DIR / img_name
            txt_path = LINES_DIR / txt_name

            text = str(row["text"])

            img = extract_image(row["image"])
            img.save(img_path)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Usually .ln files contain image paths, one per line.
            # If your code expects txt paths instead, change this to txt_path.
            ln_file.write(f"lines/{img_name}\n")

            labels[f"lines/{img_name}"] = text

            global_idx += 1


with open(LINES_DIR / "labels.pkl", "wb") as f:
    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done. Exported {global_idx} samples.")
print(f"Created: train.ln, val.ln, test.ln, lines/labels.pkl")