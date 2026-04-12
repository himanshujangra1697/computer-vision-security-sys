import numpy as np
from collections import deque
import json
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), '../../../config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

MIN_AREA = config['INTENT_THRESHOLDS']['MIN_AREA']
MAX_VARIANCE = config['INTENT_THRESHOLDS']['MAX_X_VARIANCE']
HISTORY_LEN = config['INTENT_THRESHOLDS']['HISTORY_FRAMES']

# Rolling buffer for the X-coordinate
x_history = deque(maxlen=HISTORY_LEN)

def check_intent(x, y, w, h):
    """
    Evaluates bounding box to determine if the person is approaching the door.
    Returns: (Boolean Intent, String Status Message)
    """
    area = w * h
    center_x = x + (w // 2)
    
    # Update history
    x_history.append(center_x)

    # 1. Z-Axis Check (Vicinity)
    if area < MIN_AREA:
        return False, f"Too Far (Area: {area})"

    # 2. X-Axis Check (Trajectory)
    if len(x_history) == HISTORY_LEN:
        variance = np.std(x_history)
        if variance > MAX_VARIANCE:
            return False, f"Walking Past (Var: {variance:.1f})"

    return True, "Approaching"

def clear_intent_history():
    """Clears the history buffer after a successful unlock."""
    x_history.clear()