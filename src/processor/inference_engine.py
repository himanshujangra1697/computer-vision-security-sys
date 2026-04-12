import cv2
import json
import os
import pickle
import platform
import time
import traceback
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import tensorflow as tf
from deepface import DeepFace

from processor.utils import hardware_utils
from processor.utils import intent_utils
from processor.utils.classifier_svm_utils import FaceClassifier
from processor.utils.tracker_utils import MultiFaceTracker


class ModelUpdateHandler(FileSystemEventHandler):
    def __init__(self, engine_instance):
        self.engine = engine_instance

    def on_modified(self, event):
        # We normalize the path slashes to ensure it works perfectly on Windows
        if event.src_path.replace('\\', '/').endswith("face_embeddings.pkl"):
            print("\n[SYSTEM] 🔄 New training data detected! Hot-reloading model...")
            self.engine.reload_model()

class HaloCoreEngine:
    def __init__(self):
        """
        Initialize GPU/runtime configuration, recognition models, and camera state.
        """
        print("[HALOCORE] Initializing system engine...")

        self.base_dir = os.path.dirname(__file__)
        self.config = self._load_config()

        self.blur_threshold = int(self.config["RECOGNITION"]["BLUR_THRESHOLD"])
        self.ml_threshold = float(self.config["RECOGNITION"]["ML_THRESHOLD"])
        self.cosine_threshold = float(self.config["RECOGNITION"]["COSINE_THRESHOLD"])
        
        self.unlock_cooldown_sec = max(
            1.0, float(self.config["HARDWARE"]["UNLOCK_HOLD_TIME_SEC"])
        )
        self.last_unlock_time = 0.0
        self.last_unlocked_identity = None

        self._configure_gpu()
        self.face_classifier = self._load_face_classifier()

        # --- START THE HOT-RELOAD WATCHER ---
        self.face_embeddings_path = os.path.join(self.base_dir, "face_embeddings.pkl")
        self.event_handler = ModelUpdateHandler(self)
        self.observer = Observer()
        
        # Tell the observer to watch the processor directory for file changes
        self.observer.schedule(self.event_handler, path=self.base_dir, recursive=False)
        self.observer.start()
        # ------------------------------------

        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)

        self.tracker = MultiFaceTracker()

    def _load_config(self):
        config_path = os.path.abspath(
            os.path.join(self.base_dir, "..", "..", "config.json")
        )
        with open(config_path, "r") as config_file:
            return json.load(config_file)

    def _configure_gpu(self):
        print(f"[HALOCORE] System: {platform.system()} | Processor: {platform.machine()}")
        print(f"[HALOCORE] TensorFlow Version: {tf.__version__}")

        gpus = tf.config.list_physical_devices("GPU")

        if platform.system() == "Darwin":
            if gpus:
                print(f"[HALOCORE] Apple Metal GPU enabled: {len(gpus)} GPU(s)")
            else:
                print("[HALOCORE] No GPU detected on Mac.")
                print("[HALOCORE] Install tensorflow-metal if GPU acceleration is expected.")
        elif platform.system() == "Windows":
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"[HALOCORE] Windows GPU enabled: {len(gpus)} GPU(s)")
                except RuntimeError as exc:
                    print(f"[HALOCORE] GPU configuration error: {exc}")
            else:
                print("[HALOCORE] No GPU detected on Windows.")
        else:
            if gpus:
                print(f"[HALOCORE] Linux GPU enabled: {len(gpus)} GPU(s)")

    def _load_face_classifier(self):
        face_embeddings_path = os.path.join(self.base_dir, "face_embeddings.pkl")

        with open(face_embeddings_path, "rb") as face_db_file:
            face_db = pickle.load(face_db_file)

        print("[HALOCORE] Face database loaded successfully.")

        face_classifier = FaceClassifier()
        face_classifier.train(face_db)
        return face_classifier

    @staticmethod
    def _normalize_box(frame_shape, facial_area):
        frame_height, frame_width = frame_shape[:2]

        x = max(0, int(facial_area["x"]))
        y = max(0, int(facial_area["y"]))
        w = int(facial_area["w"])
        h = int(facial_area["h"])

        x2 = min(frame_width, x + w)
        y2 = min(frame_height, y + h)
        clipped_w = x2 - x
        clipped_h = y2 - y

        if clipped_w <= 0 or clipped_h <= 0:
            return None

        return x, y, clipped_w, clipped_h

    @staticmethod
    def _is_unknown_identity(name):
        return name in {
            "Unknown",
            "background",
            "Unknown (Background)",
            "Unknown (< ml_threshold)",
        }

    def _should_unlock(self):
        now = time.monotonic()
        if now - self.last_unlock_time < self.unlock_cooldown_sec:
            return False

        self.last_unlock_time = now
        return True
    
    def reload_model(self):
        """Silently updates the SVM in RAM without crashing the video feed."""
        try:
            # Brief pause to ensure the OS has completely finished writing the new file
            time.sleep(1.0) 
            
            with open(self.face_embeddings_path, "rb") as face_db_file:
                new_face_db = pickle.load(face_db_file)
            
            # Create a brand new classifier in the background
            new_classifier = FaceClassifier()
            new_classifier.train(new_face_db)
            
            # Atomic swap: instantly replace the active classifier with the new one
            self.face_classifier = new_classifier
            print("[SYSTEM] ✅ Hot-reload complete. New identities are active.")
            
        except Exception as e:
            print(f"[SYSTEM ERROR] Failed to reload model: {e}")

    def run(self):
        """
        Continuously detect faces, evaluate intent, identify the primary subject,
        and trigger the relay for high-confidence known identities.
        """
        print("[HALOCORE] Engine online. Monitoring access point...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            draw_frame = frame.copy()

            try:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl_channel = clahe.apply(l_channel)

                enhanced_lab = cv2.merge((cl_channel, a_channel, b_channel))
                enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                embedding_objs = DeepFace.represent(
                    img_path=enhanced_frame,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=False,
                )

                if isinstance(embedding_objs, dict):
                    embedding_objs = [embedding_objs]

                current_faces_data = []
                raw_name_lookup = {}
                primary_box = None
                primary_status = None

                valid_detections = []
                for obj in embedding_objs:
                    facial_area = obj.get("facial_area")
                    if not facial_area:
                        continue

                    normalized_box = self._normalize_box(frame.shape, facial_area)
                    if not normalized_box:
                        continue

                    _, _, w, h = normalized_box
                    if w > frame.shape[1] * 0.9:
                        continue

                    valid_detections.append((normalized_box, obj))

                if valid_detections:
                    primary_box = max(
                        (box for box, _ in valid_detections),
                        key=lambda face_box: face_box[2] * face_box[3],
                    )
                else:
                    print("[HALOCORE] No faces detected in this frame.")

                for (x, y, w, h), obj in valid_detections:
                    face_img = frame[y : y + h, x : x + w]
                    if face_img.size == 0:
                        continue

                    if self.tracker.is_blurry(face_img, threshold=self.blur_threshold):
                        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                        cv2.putText(
                            draw_frame,
                            "Too Blurry",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                        )
                        continue

                    if primary_box == (x, y, w, h):
                        has_intent, primary_status = intent_utils.check_intent(x, y, w, h)
                        if not has_intent:
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            cv2.putText(
                                draw_frame,
                                primary_status,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            continue

                    embedding = obj["embedding"]
                    raw_name, score = self.face_classifier.predict(
                        embedding,
                        cosine_threshold=self.cosine_threshold,
                        ml_threshold=self.ml_threshold,
                    )

                    final_name = "Unknown (Background)" if raw_name == "background" else raw_name
                    current_faces_data.append((x, y, w, h, final_name, score))
                    raw_name_lookup[(x, y, w, h)] = raw_name

                stable_faces = self.tracker.update(current_faces_data)
                authorized_primary_identity = None

                for x, y, w, h, stable_name, score in stable_faces:
                    if self._is_unknown_identity(stable_name):
                        color = (0, 0, 255)
                        label = "UNKNOWN"
                    else:
                        color = (0, 255, 0)
                        label = stable_name.upper()

                    raw_name = raw_name_lookup.get((x, y, w, h), stable_name)
                    print(
                        f"[HALOCORE] Surveillance event -> ID: '{label}' "
                        f"(Raw: {raw_name}), Confidence: {score:.2f}"
                    )

                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        draw_frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                    if primary_box == (x, y, w, h) and primary_status:
                        cv2.putText(
                            draw_frame,
                            primary_status,
                            (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    # Removed the redundant score >= self.confidence_threshold check here
                    if (
                        primary_box == (x, y, w, h)
                        and not self._is_unknown_identity(stable_name)
                    ):
                        authorized_primary_identity = stable_name

                        if (
                            stable_name != self.last_unlocked_identity
                            and self._should_unlock()
                        ):
                            print(f"[HALOCORE] Access granted: {stable_name} ({score:.2f})")
                            hardware_utils.unlock_door()
                            intent_utils.clear_intent_history()
                            self.last_unlocked_identity = stable_name

                if authorized_primary_identity is None:
                    self.last_unlocked_identity = None

            except Exception as exc:
                print(f"[HALOCORE] Error in loop: {exc}")
                traceback.print_exc()

            cv2.imshow("HaloCore Real-time Access", draw_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()