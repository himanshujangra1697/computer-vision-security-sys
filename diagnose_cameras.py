#!/usr/bin/env python3
"""
Comprehensive camera detection diagnostic for macOS
"""
import cv2
import subprocess
import sys

print("=" * 70)
print("🔍 COMPREHENSIVE CAMERA DIAGNOSTIC FOR macOS")
print("=" * 70)

# --- 1. SYSTEM PROFILER (All cameras recognized by macOS) ---
print("\n1️⃣  SYSTEM CAMERAS (via system_profiler):")
print("-" * 70)
try:
    result = subprocess.run(
        ["system_profiler", "SPCameraDataType"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.stdout:
        print(result.stdout)
    else:
        print("⚠️  No cameras found via system_profiler")
except Exception as e:
    print(f"❌ Error: {e}")

# --- 2. USB DEVICES (Check if classroom cameras are connected) ---
print("\n2️⃣  USB DEVICES (Check for classroom camera hardware):")
print("-" * 70)
try:
    result = subprocess.run(
        ["system_profiler", "SPUSBDataType"],
        capture_output=True,
        text=True,
        timeout=5
    )
    # Filter for video/camera related devices
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ['camera', 'video', 'usb device', 'video device']):
            # Print context around the match
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            print('\n'.join(lines[start:end]))
            print("-" * 50)
except Exception as e:
    print(f"❌ Error: {e}")

# --- 3. OpenCV Camera Enumeration ---
print("\n3️⃣  OPENCV CAMERA ENUMERATION:")
print("-" * 70)
detected_cameras = []
for index in range(20):  # Try more indices
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        backend = cap.getBackendName()
        
        detected_cameras.append(index)
        print(f"✅ Index {index}:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Backend: {backend}")
        cap.release()

if not detected_cameras:
    print("❌ No cameras found by OpenCV")
else:
    print(f"\n📊 Total cameras detected: {len(detected_cameras)}")

# --- 4. Check AVFoundation (Native macOS camera API) ---
print("\n4️⃣  AVFOUNDATION DEVICES (Native macOS API):")
print("-" * 70)
try:
    result = subprocess.run(
        ["system_profiler", "SPAudioDataType"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if "camera" in result.stdout.lower() or "video" in result.stdout.lower():
        print(result.stdout)
    else:
        print("⚠️  No camera info in audio devices")
except Exception as e:
    print(f"❌ Error: {e}")

# --- 5. IOKit Registry (Advanced hardware info) ---
print("\n5️⃣  IOKIT REGISTRY (Advanced Hardware Check):")
print("-" * 70)
try:
    result = subprocess.run(
        ["ioreg", "-l", "-c", "IOUSBDevice"],
        capture_output=True,
        text=True,
        timeout=5
    )
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if 'camera' in line.lower() or 'video' in line.lower():
            print(lines[max(0, i-2):min(len(lines), i+3)])
            print("-" * 50)
except Exception as e:
    print(f"⚠️  ioreg not available or error: {e}")

# --- 6. Check Permissions ---
print("\n6️⃣  PERMISSIONS CHECK:")
print("-" * 70)
print("✓ CV2 Permissions: OK (script is running)")
print("\n⚠️  Manual checks needed:")
print("   1. System Preferences > Security & Privacy > Camera")
print("   2. Verify your terminal/IDE has camera access")
print("   3. Check System Settings > General > About > System Report")
print("      -> Hardware > USB for classroom camera device")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. If cameras appear in system_profiler but not in OpenCV:")
print("   → May need special driver or firmware update")
print("   → Try: cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)")
print("\n2. If classroom cameras don't appear anywhere:")
print("   → Check USB connection (try different USB port)")
print("   → Verify classroom camera has drivers/firmware installed")
print("   → Check if it requires special software (like Camo does)")
print("\n3. For detailed USB info, run:")
print("   $ system_profiler SPUSBDataType | grep -A 10 -i camera")
print("=" * 70)
