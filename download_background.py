import os
import requests
import tarfile
from io import BytesIO
import shutil

# Config
TARGET_DIR = "data/raw/faces/background"
LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
NUM_IMAGES_TO_KEEP = 200

def download_and_extract_background():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    print("⬇️ Downloading random face dataset (LFW)... this might take a minute...")
    response = requests.get(LFW_URL, stream=True)
    
    if response.status_code == 200:
        print("📦 Extracting images...")
        with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar:
            count = 0
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".jpg"):
                    # We flatten the structure: put all jpgs directly in background folder
                    f = tar.extractfile(member)
                    with open(os.path.join(TARGET_DIR, f"random_{count}.jpg"), 'wb') as out:
                        out.write(f.read())
                    count += 1
                    if count >= NUM_IMAGES_TO_KEEP:
                        break
        print(f"✅ Successfully saved {count} background images to '{TARGET_DIR}'")
    else:
        print("❌ Failed to download dataset.")

if __name__ == "__main__":
    download_and_extract_background()