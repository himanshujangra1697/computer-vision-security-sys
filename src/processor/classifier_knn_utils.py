import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class FaceClassifier:
    def __init__(self):
        # K=5: Look at the 5 closest photos to make a decision
        # metric='euclidean': The standard distance measure for ArcFace
        self.clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.encoder = LabelEncoder()
        self.is_trained = False

    def train(self, face_database):
        """
        Trains the KNN classifier.
        """
        X = []
        y = []

        for name, embeddings in face_database.items():
            for emb in embeddings:
                X.append(emb)
                y.append(name)

        if len(set(y)) < 2:
            print("⚠️ Warning: Need at least 2 people to train.")
            self.is_trained = False
            return

        X = np.array(X)
        y = np.array(y)

        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)

        self.clf.fit(X, y_encoded)
        self.is_trained = True
        print(f"✅ KNN Classifier trained on {len(X)} embeddings.")

    def predict(self, embedding, threshold=9.0):
        """
        Predicts identity using rigorous distance checking.
        
        distance_threshold: The maximum allowed distance. 
                            LOWER = Stricter (More Unknowns)
                            HIGHER = Looser (Fewer Unknowns)
        """
        if not self.is_trained:
            return "Unknown", 0.0

        embedding = np.array(embedding).reshape(1, -1)
        
        # 1. Find the nearest neighbors and their distances
        distances, indices = self.clf.kneighbors(embedding)
        
        # Average distance to the 5 closest matches
        avg_dist = np.mean(distances[0])
        
        # DEBUG: Print this to tune your threshold!
        print(f"   -> KNN Distance: {avg_dist:.4f}")

        # 2. STRICT CHECK: If the face is too far away, it's Unknown.
        if avg_dist > threshold:
            return "Unknown", 0.0

        # 3. Otherwise, predict the name
        prediction_index = self.clf.predict(embedding)[0]
        predicted_name = self.encoder.inverse_transform([prediction_index])[0]
        
        # Convert distance to a fake "confidence" (just for display)
        # 1.0 is perfect match, 0.0 is far away
        confidence = max(0, 1.0 - (avg_dist / 2))
        
        return predicted_name, confidence