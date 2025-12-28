# src/create_database.py
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from deepface import DeepFace

# --- CONFIGURATION ---
DATA_PATH = "../data/raw/faces"
OUTPUT_PATH = "processor/face_embeddings.pkl"
MODEL_NAME = "ArcFace"
GALLERY_SIZE = 973
# -------------------

def create_embedding_database():
    employee_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

    # This will be our new, clean database
    final_database = {}

    for name in tqdm(employee_folders, desc="Processing Employees"):
        employee_dir = os.path.join(DATA_PATH, name)

        # CRUCIAL: This list is re-initialized for every person, preventing leaks.
        current_person_embeddings = []

        image_files = [f for f in os.listdir(employee_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in tqdm(image_files, desc=f"  -> {name}", leave=False):
            image_path = os.path.join(employee_dir, image_file)
            try:
                embedding_obj = DeepFace.represent(
                    img_path=image_path,
                    model_name=MODEL_NAME,
                    enforce_detection=True
                )
                current_person_embeddings.append(embedding_obj[0]["embedding"])
            except Exception as e:
                print(f"\nWarning: Skipping {image_file}. Reason: {e}")

        if not current_person_embeddings:
            print(f"\n❌ Error: No embeddings generated for {name}. Please check images.")
            continue

        # Select a random sample for the gallery
        if len(current_person_embeddings) > GALLERY_SIZE:
            selected_embeddings = random.sample(current_person_embeddings, GALLERY_SIZE)
        else:
            selected_embeddings = current_person_embeddings

        # Add the clean gallery to our final database
        final_database[name] = selected_embeddings
        print(f"\n✅ Created gallery for {name} with {len(selected_embeddings)} entries.")

    # Save the final, correct database
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(final_database, f)

    print(f"\n🎉 New database created successfully at '{OUTPUT_PATH}'")

if __name__ == "__main__":
    create_embedding_database()