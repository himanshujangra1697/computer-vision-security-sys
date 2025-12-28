import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class FaceClassifier:
    def __init__(self):
        # Linear kernel is usually best for high-dim embeddings like ArcFace
        self.clf = SVC(kernel='linear', probability=True)
        self.encoder = LabelEncoder()

        # Centroids for the "Gatekeeper" check
        self.centroids = {}
        self.is_trained = False

    def train(self, face_database):
        """
        Trains BOTH the SVM (for precision) and Centroids (for gating).
        """
        X = []
        y = []
        self.centroids = {}

        # Flatten the database (dictionary of galleries) into lists
        for name, embeddings in face_database.items():
            if not embeddings: continue
            
            # --- CENTROID LOGIC ---
            # Calculate the average face for this person
            emb_array = np.array(embeddings)
            centroid = np.mean(emb_array, axis=0)
            # Normalize it
            norm = np.linalg.norm(centroid)
            if norm > 0: centroid = centroid / norm
            self.centroids[name] = centroid
            
            for emb in embeddings:
                X.append(emb)
                y.append(name)

        # Safety check: Need at least 2 classes to train
        if len(set(y)) < 2:
            print("⚠️ Classifier Warning: Need at least 2 different people in DB to train.")
            self.is_trained = False
            return

        X = np.array(X)
        y = np.array(y)

        # Encode names (strings) into numbers
        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)

        # Train the SVM
        self.clf.fit(X, y_encoded)
        self.is_trained = True
        print(f"""✅ Classifier trained successfully on {len(X)} total face embeddings.\n
                ✅ Hybrid Classifier trained (SVM + {len(self.centroids)} Centroids).""")

    def predict(self, embedding, gate_threshold=0.35):
        """
        Hybrid Prediction:
        1. GATE: Check Cosine Similarity to Centroids. If too far -> Unknown.
        2. JUDGE: If passed, ask SVM for precise name.
        """
        if not self.is_trained: return "Unknown", 0.0

        embedding = np.array(embedding).reshape(1, -1)
        
        # --- STEP 1: THE GATEKEEPER (Cosine) ---
        best_cosine_score = -1.0
        
        for name, centroid in self.centroids.items():
            if name == "background": continue # Don't match to background centroid
            
            centroid = centroid.reshape(1, -1)
            score = cosine_similarity(embedding, centroid)[0][0]
            
            if score > best_cosine_score:
                best_cosine_score = score
        
        # REJECTION LOGIC: If you don't look like the average of ANYONE, you are Unknown.
        # This blocks random strangers immediately.
        if best_cosine_score < gate_threshold:
            return "Unknown", best_cosine_score
        
        # --- STEP 2: THE JUDGE (SVM) ---
        # If we get here, you look somewhat like someone. Let SVM decide exactly who.
        probs = self.clf.predict_proba(embedding)[0]
        best_class_index = np.argmax(probs)
        svm_score = probs[best_class_index]
        predicted_name = self.encoder.inverse_transform([best_class_index])[0]
        
        # If SVM is unsure, fallback to Unknown
        if svm_score < 0.75: # SVM needs high confidence
            return "Unknown", svm_score
        
        return predicted_name, svm_score