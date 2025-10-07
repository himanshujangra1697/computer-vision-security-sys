import cv2
import os
import time

# --- CONFIGURATION ---
# Directory to save the images
SAVE_PATH = "data/raw/faces"
# Number of images to capture
IMAGES_TO_COLLECT = 30
# --- END CONFIGURATION ---

def collect_face_data():
    """
    Detects a face from the webcam feed and saves images upon key press.
    """
    employee_name = input("Enter the employee's name (no spaces, e.g., 'himanshu_jangra'): ").lower()
    save_dir = os.path.join(SAVE_PATH, employee_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory created: {save_dir}")

    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture from the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    img_counter = 0
    while img_counter < IMAGES_TO_COLLECT:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to grayscale for the face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw a rectangle around the detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display instructions and status on the frame
        status_text = f"Images captured: {img_counter}/{IMAGES_TO_COLLECT}"
        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the live feed
        cv2.imshow("Face Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s') and len(faces) > 0:
            # Save the captured frame if a face is detected
            img_name = f"{employee_name}_{int(time.time())}.jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"✅ Image saved: {img_name}")
            img_counter += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCollection finished. {img_counter} images saved in {save_dir}")

if __name__ == "__main__":
    collect_face_data()