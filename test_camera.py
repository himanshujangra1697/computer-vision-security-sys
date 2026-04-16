import cv2

def list_camera_devices():
    """Find all available camera devices"""
    available_cameras = []
    
    # Try indices 0-10
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            name = f"Camera {index}"
            available_cameras.append((index, name))
            cap.release()
            print(f"✅ Found: Camera {index}")
    
    if not available_cameras:
        print("❌ No cameras found")
    else:
        print(f"\n📷 Total cameras found: {len(available_cameras)}")
    
    return available_cameras

if __name__ == "__main__":
    cameras = list_camera_devices()