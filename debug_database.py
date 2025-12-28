import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "src/processor/face_embeddings.pkl"

def analyze_database():
    try:
        with open(DB_PATH, "rb") as f:
            face_db = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Database not found at {DB_PATH}")
        return

    people = list(face_db.keys())
    if len(people) < 2:
        print("❌ Error: Database must contain at least two people to compare.")
        return

    person1_name, person2_name = people[0], people[1]
    gallery1 = np.array(face_db[person1_name])
    gallery2 = np.array(face_db[person2_name])

    print("\n--- Database Analysis ---")
    print(f"Comparing '{person1_name}' with '{person2_name}'...")
    print(f"Gallery size for {person1_name}: {len(gallery1)}")
    print(f"Gallery size for {person2_name}: {len(gallery2)}")

    # 1. Self-Similarity (should be high)
    # Compare the first embedding of person1 to their own gallery
    self_similarity_scores = cosine_similarity(gallery1[0].reshape(1, -1), gallery1)
    avg_self_similarity = np.mean(self_similarity_scores)
    print(f"\nAverage self-similarity for {person1_name}: {avg_self_similarity:.2f} (should be high, > 0.8)")

    # 2. Cross-Similarity (should be low)
    # Compare the first embedding of person1 to person2's gallery
    cross_similarity_scores = cosine_similarity(gallery1[0].reshape(1, -1), gallery2)
    avg_cross_similarity = np.mean(cross_similarity_scores)
    print(f"Average cross-similarity between {person1_name} and {person2_name}: {avg_cross_similarity:.2f} (should be low, < 0.5)")

    if avg_cross_similarity > 0.8:
        print("\n🚨 DIAGNOSIS: CRITICAL FAILURE! The galleries are nearly identical. This confirms a bug in the database creation process.")
    else:
        print("\n✅ DIAGNOSIS: Galleries appear distinct. The issue may lie elsewhere.")
    print("-------------------------\n")


if __name__ == "__main__":
    analyze_database()