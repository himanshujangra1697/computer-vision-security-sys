#!/usr/bin/env python3
import cv2
import time

print("=" * 60)
print("Testing all camera indices (0-20) with different backends")
print("=" * 60)

# Try with default backend
print("\n1️⃣  DEFAULT BACKEND:")
print("-" * 60)
for index in range(21):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Try to read a frame to verify it's really working
        ret, frame = cap.read()
        
        print(f"✅ Index {index}: {width}x{height} @ {fps}fps (Frame read: {ret})")
        cap.release()

# Try with AVFoundation backend explicitly
print("\n2️⃣  AVFOUNDATION BACKEND (cv2.CAP_AVFOUNDATION):")
print("-" * 60)
for index in range(21):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        ret, frame = cap.read()
        
        print(f"✅ Index {index}: {width}x{height} @ {fps}fps (Frame read: {ret})")
        cap.release()

print("\n" + "=" * 60)
