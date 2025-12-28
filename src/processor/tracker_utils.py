from collections import deque, Counter
import numpy as np

class FaceTracker:
    """Maintains history for ONE specific person (ID)."""
    def __init__(self, buffer_size=5):
        self.prediction_history = deque(maxlen=buffer_size)
        self.missed_frames = 0  # Counts how many frames this person has been gone
        
    def update(self, name):
        """Add a new prediction to the history."""
        self.prediction_history.append(name)
        self.missed_frames = 0  # Reset missed frames count

    def get_stable_prediction(self):
        """Returns the most common name in the history buffer."""
        if not self.prediction_history: return "Unknown"
            
        # Count the frequency of names in the buffer
        counts = Counter(self.prediction_history)
        most_common_name, count = counts.most_common(1)[0]
        
        # QUALITY GATE:
        # Only return the name if it appears in > 60% of the last frames
        # Otherwise, the system is unsure -> return Unknown
        if count / len(self.prediction_history) > 0.6:
            return most_common_name
        else:
            return "Unknown"


class MultiFaceTracker:
    """Manages multiple FaceTrackers based on screen location."""
    def __init__(self):
        self.trackers = {} # Dictionary: { internal_id: FaceTracker() }
        self.centers = {}  # Dictionary: { internal_id: (x, y) }
        self.next_id = 0

    def update(self, current_faces):
        """
        Input: List of (x, y, w, h, predicted_name)
        Output: List of (x, y, w, h, stable_name)
        """
        active_ids = []
        results = []

        for (x, y, w, h, name, score) in current_faces:
            cx, cy = x + w // 2, y + h // 2
            
            # 1. Find the closest existing tracker
            best_id = -1
            min_dist = 99999
            
            for tid, center in self.centers.items():
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(center))
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

            # 2. Logic: If closer than 100 pixels, it's the same person
            if min_dist < 100 and best_id != -1 and best_id not in active_ids:
                curr_id = best_id
            else:
                # New person entered
                curr_id = self.next_id
                self.trackers[curr_id] = FaceTracker()
                self.next_id += 1

            # 3. Update that specific tracker
            self.trackers[curr_id].update(name)
            self.centers[curr_id] = (cx, cy)
            active_ids.append(curr_id)
            
            # 4. Get stable result
            stable_name = self.trackers[curr_id].get_stable_prediction()
            results.append((x, y, w, h, stable_name, score))

        # 5. Cleanup: Remove trackers for people who left the frame
        # (If we haven't seen ID #5 in this frame, mark it missed)
        current_tracker_ids = list(self.trackers.keys())
        for tid in current_tracker_ids:
            if tid not in active_ids:
                self.trackers[tid].missed_frames += 1
                # If gone for 30 frames (1 second), delete memory
                if self.trackers[tid].missed_frames > 30:
                    del self.trackers[tid]
                    del self.centers[tid]

        return results

    def is_blurry(self, face_img, threshold=100):
        """
        Returns True if the face is too blurry to trust.
        Uses Laplacian Variance method.
        """
        import cv2

        try:
            target_size = (112, 112)
            resized_face = cv2.resize(face_img, target_size)

            gray = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            enhanced_gray = cv2.equalizeHist(gray)

            score = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
            # If score < threshold, image is blurry
            return score < threshold
        except:
            return True