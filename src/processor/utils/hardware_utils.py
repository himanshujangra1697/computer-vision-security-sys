import serial
import time
import json
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), '../../../config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Initialize Relay
RELAY_ENABLED = config['HARDWARE']['ENABLE_RELAY']
PORT = config['HARDWARE']['COM_PORT']
BAUD = config['HARDWARE']['BAUD_RATE']
HOLD_TIME = config['HARDWARE']['UNLOCK_HOLD_TIME_SEC']

relay = None

if RELAY_ENABLED:
    try:
        relay = serial.Serial(PORT, BAUD, timeout=1)
        print(f"[HARDWARE] USB Relay connected on {PORT}")
    except Exception as e:
        print(f"[HARDWARE WARNING] Could not connect to relay on {PORT}. Running in software-only mode.")
        RELAY_ENABLED = False

def unlock_door():
    """Sends the hex command to trigger the physical relay."""
    if RELAY_ENABLED and relay and relay.is_open:
        print("[HARDWARE] 🟢 LOCK OPEN")
        relay.write(b'\xA0\x01\x01\xA2')  # Turn Relay ON
        time.sleep(HOLD_TIME)
        relay.write(b'\xA0\x01\x00\xA1')  # Turn Relay OFF
        print("[HARDWARE] 🔴 LOCK CLOSED")
    else:
        print("[SOFTWARE MOCK] 🟢 DOOR UNLOCKED (Hardware disabled)")
        time.sleep(HOLD_TIME)