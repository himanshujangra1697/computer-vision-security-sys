import cv2
import pickle
import traceback
import platform
import sys
import tensorflow as tf
from deepface import DeepFace
from classifier_svm_utils import FaceClassifier
from tracker_utils import MultiFaceTracker

# --- 1. ROBUST CROSS-PLATFORM GPU SETUP ---
# This block handles the differences between Mac M-Chips and Windows NVIDIA
print(f"🖥️  System: {platform.system()} | Processor: {platform.machine()}")
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')

if platform.system() == 'Darwin':
    # --- MAC OS (M1/M2/M3/M4) ---
    if gpus:
        print(f"✅ Apple Metal GPU Enabled: {len(gpus)} GPU(s)")
        # Metal manages memory automatically, so we don't need set_memory_growth
    else:
        print("⚠️  No GPU detected on Mac.")
        print("   -> Run: pip install tensorflow-metal")

elif platform.system() == 'Windows':
    # --- WINDOWS ---
    if gpus:
        try:
            # specific optimization to prevent locking all VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Windows NVIDIA GPU Enabled: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"❌ GPU Error: {e}")
    else:
        print("⚠️  No GPU detected on Windows.")
        print("   -> Ensure you installed TensorFlow 2.10 and CUDA 11.2")

else:
    # --- LINUX / OTHER ---
    if gpus:
        print(f"✅ Linux GPU Enabled: {len(gpus)} GPU(s)")
# -----------------------------------------------------------

# --- CONFIG ---
BLUR_THRESHOLD = 50
ML_THRESHOLD = 0.65
COSINE_THRESHOLD = 0.600
# ----------------

# --- SETUP ---
print("--- STARTING PRODUCTION SYSTEM ---")
try:
    with open("face_embeddings.pkl", "rb") as f:
        face_db = pickle.load(f)
    print("✅ Face database loaded successfully.")
except FileNotFoundError:
    print("❌ Error: face_embeddings.pkl not found. Please run create_database.py first.")
    exit()

# --- Initialize and Train the Classifier ---
face_classifier = FaceClassifier()
face_classifier.train(face_db)

# --- CROSS-PLATFORM CAMERA SETUP ---
# Keep this logic! It is crucial for Windows performance.
if platform.system() == 'Windows':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Faster on Windows
else:
    cap = cv2.VideoCapture(0) # Standard for Mac (AVFoundation)
# -----------------------------------

# Initialize the Stabilizer
tracker = MultiFaceTracker()

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # Make a copy for drawing the final boxes
    draw_frame = frame.copy()
    
    try:

        # --- OPTION 2: CLAHE PRE-PROCESSING (Lighting Fixer) ---
        # This enhances local contrast to fix shadows/glare
        
        # 1. Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # 3. Merge back
        limg = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # -------------------------------------------------------

        # --- 1. FACE DETECTION & EMBEDDING EXTRACTION ---
        # Note: 'retinaface' is best, but if it lags, try 'yunet' or 'opencv' for speed
        # USE 'enhanced_frame' for detection now
        embedding_objs = DeepFace.represent(
            img_path = enhanced_frame,
            model_name = "ArcFace",
            detector_backend = 'retinaface',
            enforce_detection = False
        )

        if len(embedding_objs) == 0:
            print("No faces detected in this frame.")

        # 3. GATHER RAW DATA
        current_faces_data = [] # Stores (x,y,w,h, name) for everyone
        
        # Loop through EACH detected face independently
        for obj in embedding_objs:
            # Check if this is a "dummy" detection (DeepFace sometimes returns full image if no face found)
            facial_area = obj["facial_area"]
            # x, y, w, h = facial_area.values()
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']
            # If the box is the size of the whole screen, it's a false detection
            if w > frame.shape[1] * 0.9: continue

            # --- 2. EXTRACT THE FACE & CHECK BLURRINESS ---
            # Extract the actual face image
            face_img = frame[y:y+h, x:x+w]
            
            # If the face is blurry, skip prediction to prevent bad data
            if tracker.is_blurry(face_img, threshold=BLUR_THRESHOLD):
                cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 165, 255), 2) # Orange = Blurry
                cv2.putText(draw_frame, "Too Blurry", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                continue
            
            # --- 3. PREDICT ---
            embedding = obj["embedding"]
            name, score = face_classifier.predict(embedding, cosine_threshold=COSINE_THRESHOLD, ml_threshold=ML_THRESHOLD)
            
            final_name = name

            # If the model thinks it's the wall/background, treat as Unknown
            if name == "background":
                final_name = "Unknown (Background)"

            # Store in list
            current_faces_data.append((x, y, w, h, final_name, score))

        # 4. TRACK & STABILIZE (The Magic Step)
        # The tracker sorts out which name belongs to which face location
        stable_faces = tracker.update(current_faces_data)

        # 5. DRAW FINAL RESULTS
        for (x, y, w, h, stable_name, score) in stable_faces:            
            # --- 6. DISPLAY LOGIC ---
            if (stable_name == "Unknown") or (stable_name == "background") or (stable_name == "Unknown (Background)"):
                color = (0, 0, 255) # Red for Unknown
                label = "UNKNOWN"
            else:
                color = (0, 255, 0) # Green for Known
                label = stable_name.upper()

            print(f"Surveillance Event -> ID: '{label}' (Raw: {final_name}), Confidence: {score:.2f}")

            cv2.rectangle(draw_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(draw_frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # cv2.putText(draw_frame, f"{label} ({score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    except Exception as e:
        print(f"⚠️ Error in loop: {e}")
        traceback.print_exc()
        pass

    # Show the Draw Frame (which is based on the original colored frame)
    cv2.imshow('Real-time Access Monitor', draw_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()