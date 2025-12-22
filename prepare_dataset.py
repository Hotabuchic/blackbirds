import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# --------- НАСТРОЙКИ ----------
INPUT_CSV = "observations_filtered.csv"
OUTPUT_CSV = "birds_3000.csv"
IMAGES_DIR = Path("data/images")
N_SAMPLES = 3000
# -----------------------------

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Загружаем CSV
df = pd.read_csv(INPUT_CSV)

# Берём первые 3000 строк
df = df.head(N_SAMPLES)

records = []

for idx, row in df.iterrows():
    image_url = row["image_url"]
    common_name = row["common_name"]
    sample_id = row["id"]

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")

        image_path = IMAGES_DIR / f"{sample_id}.jpg"
        img.save(image_path, quality=85, optimize=True)

        records.append({
            "id": sample_id,
            "image_url": image_url,
            "common_name": common_name,
            "image_path": str(image_path)
        })

        if len(records) % 100 == 0:
            print(f"Downloaded {len(records)} images")

    except Exception as e:
        print(f"Failed to download {image_url}: {e}")

# Сохраняем новый датасет
new_df = pd.DataFrame(records)
new_df.to_csv(OUTPUT_CSV, index=False)

print("\nDone!")
print(f"Saved images: {len(new_df)}")
print(f"New dataset CSV: {OUTPUT_CSV}")
