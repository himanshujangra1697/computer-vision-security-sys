import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class FaceClassifier:
    def __init__(self):
        # Linear kernel is usually best for high-dim embeddings like ArcFace
        self.clf = SVC(kernel='linear', probability=True)
        self.encoder = LabelEncoder()

        # CHANGED: Store raw embeddings (face_memory) instead of Centroids
        self.face_memory = {} 
        self.is_trained = False

    def train(self, face_database):
        """
        Trains BOTH the SVM (for precision) and stores Raw Embeddings (for Best-Match Gating).
        """
        X = []
        y = []
        self.face_memory = {}

        # Flatten the database (dictionary of galleries) into lists
        for name, embeddings in face_database.items():
            if not embeddings: continue
            
            # --- NEW: BEST MATCH LOGIC ---
            # Store ALL embeddings for this person as a matrix
            # This allows us to find the single closest photo later
            self.face_memory[name] = np.array(embeddings)
            
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
        print(f"✅ Hybrid Classifier trained (SVM + Best-Match Gate on {len(X)} images).")

    def predict(self, embedding, cosine_threshold=0.40, ml_threshold=0.60):
        """
        Hybrid Prediction (Upgraded):
        1. GATE: Find the SINGLE closest photo in the database (Max Similarity).
        2. JUDGE: If passed, ask SVM for precise name.
        """
        if not self.is_trained: return "Unknown", 0.0

        embedding = np.array(embedding).reshape(1, -1)
        
        best_gate_score = -1.0
        potential_gate_name = "Unknown"

        # --- STEP 1: THE "BEST MATCH" GATEKEEPER ---
        # Compare live face against ALL photos of EACH person
        for name, db_embeddings in self.face_memory.items():
            if name == "background": continue # Don't match to background
            
            # Calculate similarity to ALL images of this person at once
            similarities = cosine_similarity(embedding, db_embeddings)
            
            # Find the single highest score (The "Best Match")
            max_score = np.max(similarities)
            
            if max_score > best_gate_score:
                best_gate_score = max_score
                potential_gate_name = name

        if best_gate_score > cosine_threshold:
            print(f"🔍 COSINE [Level 1]: Best Match: {potential_gate_name} | Score: {best_gate_score:.3f}")
            return potential_gate_name, best_gate_score
        
        else:
            # --- STEP 2: THE JUDGE (SVM) ---
            # If we get here, you look somewhat like someone. Let SVM decide exactly who.
            probs = self.clf.predict_proba(embedding)[0]
            best_class_index = np.argmax(probs)
            svm_score = probs[best_class_index]
            predicted_name = self.encoder.inverse_transform([best_class_index])[0]
            
            # If SVM is unsure, fallback to Unknown
            if svm_score < ml_threshold: 
                print(f"⚠️ SVM [Level 2] and COSINE [Level 1], both are failed to meet thresholds.")
                return "Unknown (< ml_threshold)", svm_score

            print(f"🧐 SVM [Level 2]: Best Match: {potential_gate_name} | Score: {best_gate_score:.3f}")
            return predicted_name, svm_score