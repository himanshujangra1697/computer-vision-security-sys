# Function to handle the logic of comparing a face to our database.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_match(embedding, face_database, threshold=0.6):
    """
    Finds the closest match for a given face embedding in the database.

    Args:
        embedding (np.ndarray): The embedding of the face to recognize.
        face_database (dict): The dictionary of known face embeddings.
        threshold (float): The minimum similarity score to be considered a match.

    Returns:
        tuple: A tuple containing the name of the matched person and the similarity score.
               Returns ("Unknown", 0.0) if no match is found above the threshold.
    """
    if not face_database:
        return "Unknown", 0.0

    # Reshape the input embedding to be a 2D array for cosine_similarity
    embedding = np.array(embedding).reshape(1, -1)

    best_match_name = "Unknown"
    best_match_score = 0.0

    for name, db_embedding in face_database.items():
        # Reshape the database embedding
        db_embedding = np.array(db_embedding).reshape(1, -1)

        # Calculate similarity
        sim_score = cosine_similarity(embedding, db_embedding)[0][0]

        if sim_score > best_match_score:
            best_match_score = sim_score
            best_match_name = name

    # If the best match is still below the threshold, consider it unknown
    if best_match_score < threshold:
        return "Unknown", best_match_score

    return best_match_name, best_match_score