import cv2
import os
import time

# --- CONFIG ---
PERSON_NAME = "shivani_shahi" # Change this to the person's name
SAVE_PATH = f"../computer-vision-security-sys/data/raw/faces/{PERSON_NAME}"
# ----------------

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

cap = cv2.VideoCapture(1) # Ensure this matches your CCTV index
if not cap.isOpened(): cap = cv2.VideoCapture(0)

print(f"📸 Camera Active. It will capture photos for {PERSON_NAME} every 2 seconds.")
print("   Capture different angles/expressions. Press 'q' to quit.")

count = 0
last_capture_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow("Calibration Mode", frame)
    key = cv2.waitKey(1) & 0xFF

    if time.time() - last_capture_time >= 2:
        # Save frame with a unique timestamp
        filename = f"cctv_calibration_{int(time.time())}_{count}.png"
        filepath = os.path.join(SAVE_PATH, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved: {filename}")
        count += 1

        last_capture_time = time.time()
    
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()