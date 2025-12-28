# Function to handle the logic of comparing a face to our database.

import numpy as np
# import cv2
from sklearn.metrics.pairwise import cosine_similarity

# def align_face(img, left_eye, right_eye):
#     """
#     Aligns a face image based on the eye positions.

#     Args:
#         img (np.ndarray): The image containing the face.
#         left_eye (tuple): The (x, y) coordinates of the left eye.
#         right_eye (tuple): The (x, y) coordinates of the right eye.

#     Returns:
#         np.ndarray: The aligned face image.
#     """
#     # Calculate the angle between the eyes
#     dY = right_eye[1] - left_eye[1]
#     dX = right_eye[0] - left_eye[0]
#     angle = np.degrees(np.arctan2(dY, dX))

#     # Get the center of the image
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)

#     # Get the rotation matrix
#     M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
#     # Apply the rotation
#     aligned_img = cv2.warpAffine(img, M, (w, h))
    
#     return aligned_img

def find_closest_match(embedding, face_database, threshold=0.65):
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

    embedding = np.array(embedding).reshape(1, -1)

    best_match_name = "Unknown"
    best_overall_score = 0.0

    # --- START: CORRECT GALLERY LOGIC ---
    # Loop through each person in the database (e.g., 'himanshu_jangra')
    for name, embedding_gallery in face_database.items():
        
        # For each person, find their single best score from their gallery of embeddings
        best_person_score = 0.0
        for db_embedding in embedding_gallery:
            db_embedding = np.array(db_embedding).reshape(1, -1)
            sim_score = cosine_similarity(embedding, db_embedding)[0][0]
            if sim_score > best_person_score:
                best_person_score = sim_score
        
        # --- DEBUG PRINT ---
        print(f"    Comparing with '{name}', Best Gallery Score: {best_person_score:.2f}")

        # Check if this person is the best overall match found so far
        if best_person_score > best_overall_score:
            best_overall_score = best_person_score
            best_match_name = name
    # --- END: CORRECT GALLERY LOGIC ---
    
    # Final check against the threshold happens AFTER checking all people
    if best_overall_score < threshold:
        return "Unknown", best_overall_score
        
    return best_match_name, best_overall_score