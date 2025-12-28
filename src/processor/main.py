import cv2
import pickle
import traceback
from deepface import DeepFace
from classifier_svm_utils import FaceClassifier
from tracker_utils import FaceTracker
from tracker_utils import MultiFaceTracker


# --- CONFIG ---
BLUR_THRESHOLD = 50   # Adjust based on your camera (Lower = allow more blur)
CONFIDENCE_THRESHOLD = 0.80 # Strict confidence required

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
# -----------------------------------------------

if cv2.VideoCapture(1):
    cap = cv2.VideoCapture(1)
else:
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

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
            x, y, w, h = facial_area.values()

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
            name, score = face_classifier.predict(embedding, gate_threshold=0.30)
            
            # --- 4. CONFIDENCE GATE ---
            # Even if SVM says "Himanshu", if prob is 55%, treat as Unknown
            if score < CONFIDENCE_THRESHOLD:
                final_name = "Unknown"
            else:
                final_name = name

            # Store in list
            current_faces_data.append((x, y, w, h, name, score))

        # 4. TRACK & STABILIZE (The Magic Step)
        # The tracker sorts out which name belongs to which face location
        stable_faces = tracker.update(current_faces_data)

        # 5. DRAW FINAL RESULTS
        for (x, y, w, h, stable_name, score) in stable_faces:            
            # --- 6. DISPLAY LOGIC ---
            if (stable_name == "Unknown") or (stable_name == "background"):
                color = (0, 0, 255) # Red for Unknown
                label = "UNKNOWN"
            else:
                color = (0, 255, 0) # Green for Known
                label = stable_name.upper()

            print(f"Surveillance Event -> ID: '{label}' (Raw: {stable_name}), Confidence: {score:.2f}")

            cv2.rectangle(draw_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(draw_frame, f"{label} ({score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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