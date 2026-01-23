# 🛡️ Computer Vision Security System (Release 1.0)

A high-performance, real-time face recognition system designed for security monitoring. It features a hybrid classification engine (SVM + Cosine Gating), anti-blur checks, and cross-platform GPU acceleration for both macOS (Apple Silicon) and Windows (NVIDIA).

## ✨ Key Features
- **Hybrid Recognition:** Combines DeepFace embeddings with an SVM classifier for high confidence.
- **Hardware Accelerated:** Runs natively on Mac Metal (M1-M4) and Windows CUDA.
- **Stabilization:** Tracks faces across frames to prevent ID flickering.
- **Quality Control:** Automatically ignores blurry images or bad lighting.

---

## 🛠️ Prerequisites

### 1. Python 3.11 (Required)
This project **requires Python 3.11** to handle dependency conflicts between Mac and Windows libraries.
- **Mac:** `brew install python@3.11` or download from [python.org](https://www.python.org/downloads/).
- **Windows:** Download Python 3.11 from [python.org](https://www.python.org/downloads/).

### 2. Windows Only: C++ Build Tools
If you are on Windows, you must install **Visual Studio C++ Build Tools** to compile `dlib`.
1. Download from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. During install, select **"Desktop development with C++"**.

---

## 📦 Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/himanshujangra1697/computer-vision-security-sys.git
cd computer-vision-security-sys
```

### 2. Create Virtual Environment (Python 3.11)
**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install Dependencies
We use a smart requirements file that detects your OS automatically.
```bash
python -m pip install --force-reinstall "PackageName==25.2"
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run
**Step 1: Prepare Your Database**
1. Open the folder named ```data/raw/faces``` (in root folder).

2. Inside, create subfolders for each person:
   ```
   /data/raw/faces
        /Elon_Musk
            photo1.jpg
            photo2.jpg
        /Bill_Gates
            photo1.jpg
   ```
3. Create a folder name ```background``` in this same folder (for model training).
    https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
    > Refer the above link to download random images and run ```download_database_manual.py```
4. Open ```/src``` (in root folder).
5. Run ```augment_dataset.py``` -
    ```python
    # You can use it on default
    TARGET_IMAGES_PER_PERSON = 1000 # Total number of images after augmentation
    ```
    ```bash
    python augment_dataset.py
    ```
6. Create image embedding (we are using Arcface) -
    ```python
    # --- CONFIGURATION ---
    DATA_PATH = "../data/raw/faces"
    OUTPUT_PATH = "processor/face_embeddings.pkl"
    MODEL_NAME = "ArcFace"
    GALLERY_SIZE = 1000
    # -------------------
    ```
    ```bash
    python create_database.py
    ```

**Step 2: Start the System**

Open the folder named ```src/processor``` (in root folder).
>Connect your webcam and run:

```bash
python main.py
```

## 🏗️ Building for Production (Optional)
To create a standalone executable that doesn't require Python installed:

**On macOS (Creates .app bundle):**
```bash
pyinstaller build_mac.spec --noconfirm --clean
# App will be in dist/SecuritySystem.app
```

**On Windows (Creates .exe file):**
```bash
pyinstaller --onefile --windowed main.py
```

## ⚠️ Troubleshooting
- "No GPU Detected": Ensure you are using the virtual environment created with Python 3.11. Python 3.12 is not supported for GPU on Windows yet.
- Mac Camera Crash: If building a .app, ensure build_mac.spec includes the NSCameraUsageDescription plist entry.

---

>### 🔮 Release 2: The Roadmap (What's Next?)

Now that the core engine is stable, Release 2 focuses on **Security, Scalability, and User Interface.**

1.  **Liveness Detection (Anti-Spoofing):**
    * *Problem:* Currently, someone can hold up a photo of "Elon Musk" and unlock the system.
    * *Solution:* Implement "Silent Liveness" (checking for depth/texture) or "Active Liveness" (asking the user to blink or turn their head) to ensure it's a real human.

2.  **Scalable Vector Database:**
    * *Problem:* `pickle` files are slow if you have 1,000+ employees.
    * *Solution:* Migrate storage to **ChromaDB** or **FAISS**. This allows searching millions of faces in milliseconds.

3.  **Web Dashboard (FastAPI + React/Streamlit):**
    * *Problem:* `cv2.imshow` only works on the machine running the code.
    * *Solution:* Stream the video feed to a web browser so security guards can watch from an iPad or remote computer.

4.  **Instant Alert System:**
    * *Feature:* Integration with Slack, Discord, or Email.
    * *Trigger:* If an "Unknown" person lingers for >10 seconds, or a "Blacklisted" person is detected, instantly send a photo alert to the admin.

5.  **Stranger Logging:**
    * *Feature:* Automatically save a screenshot of every "Unknown" face into a `daily_logs/` folder with a timestamp, creating a digital visitor log.