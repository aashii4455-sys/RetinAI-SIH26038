import pandas as pd
from pathlib import Path

DATASET = Path("dataset/aptos2019-blindness-detection")
CSV_FILE = DATASET / "train.csv"
IMAGE_DIR = DATASET / "train_images"

print("\n=== SIH26038 DATASET CHECK ===\n")

df = pd.read_csv(CSV_FILE)

print("CSV rows:", len(df))
print("Columns:", list(df.columns))

images = list(IMAGE_DIR.glob("*"))

print("Images found:", len(images))

image_names = {p.stem for p in images}

csv_names = set(df["id_code"].astype(str))

missing_images = csv_names - image_names
extra_images = image_names - csv_names

print("Missing images:", len(missing_images))
print("Extra images:", len(extra_images))

print("\nDR CLASS DISTRIBUTION:")
print(df["diagnosis"].value_counts().sort_index())

if len(df) == len(images) and len(missing_images) == 0:
    print("\n✅ DATASET IS READY FOR TRAINING!")
else:
    print("\n⚠️ DATASET NEEDS CHECKING.")