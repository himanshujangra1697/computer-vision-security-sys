import cv2
import pickle
from deepface import DeepFace
from face_utils import find_closest_match

# --- SETUP ---
try:
    with open("face_embeddings.pkl", "rb") as f:
        face_db = pickle.load(f)
    print("✅ Face database loaded successfully.")
except FileNotFoundError:
    print("❌ Error: face_embeddings.pkl not found. Please run Phase 2 notebook first.")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # Use a fast detector. Set enforce_detection=False to avoid crashes.
        detected_faces = DeepFace.extract_faces(frame, detector_backend='ssd', enforce_detection=False)

        # --- NEW: PRINT THE DETECTED FACES OBJECT ---
        # This will show us if any faces are being detected at all.
        if detected_faces:
             print(f"Detected {len(detected_faces)} face(s).")
        # -------------------------------------------

        for face_obj in detected_faces:
            # --- NEW: PRINT THE CONFIDENCE SCORE ---
            # This is the most important clue.
            confidence = face_obj['confidence']
            print(f"Face detected with confidence: {confidence:.2f}")
            # ---------------------------------------
            
            # The current threshold is 0.90. If the printed confidence is lower, this block is skipped.
            if confidence > 0.90: 
                x, y, w, h = face_obj['facial_area'].values()
                face_img = frame[y:y+h, x:x+w] 

                embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                name, score = find_closest_match(embedding, face_db)

                print(f"Match Attempt: Best match is '{name}' with score {score:.2f}")

                if name != "Unknown":
                    color = (0, 255, 0)
                    label = f"{name} ({score:.2f})"
                else:
                    color = (0, 0, 255)
                    label = "UNKNOWN"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    except Exception as e:
        # --- MODIFIED: PRINT THE ERROR INSTEAD OF IGNORING IT ---
        print(f"An error occurred in the loop: {e}")
        # --------------------------------------------------------

    cv2.imshow('Real-time Access Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()