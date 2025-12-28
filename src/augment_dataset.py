import os
import cv2
import random
import argparse
import albumentations as A
from tqdm import tqdm

# --- CONFIGURATION ---
# The path to your dataset. This is now both the input and output directory.
DATA_PATH = "../data/raw/faces" 
# The target number of images for each person after augmentation.
TARGET_IMAGES_PER_PERSON = 1000
# --- END CONFIGURATION ---

# Define the series of augmentations to apply.
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Rotate(limit=10, p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2)
])

def augment_dataset(people_to_augment=None):
    """
    Applies augmentations to the dataset to increase its size and variety.
    Saves augmented images back into the original person's folder.

    Args:
        people_to_augment (list, optional): A list of specific people (folder names) to augment.
                                            If None, all people will be augmented. Defaults to None.
    """
    if people_to_augment:
        person_folders = [p for p in people_to_augment if os.path.isdir(os.path.join(DATA_PATH, p))]
        print(f"Targeting {len(person_folders)} specific person(s) for augmentation.")
    else:
        person_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
        print("No specific people specified. Targeting all folders for augmentation.")

    if not person_folders:
        print(f"❌ Error: No valid person folders found in '{DATA_PATH}'.")
        return

    # Loop through each person's folder
    for person_name in tqdm(person_folders, desc="Overall Progress"):
        person_dir = os.path.join(DATA_PATH, person_name)
        
        original_images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_originals = len(original_images)
        num_to_generate = TARGET_IMAGES_PER_PERSON - num_originals

        if num_to_generate <= 0:
            print(f"\nSkipping {person_name}, already has {num_originals} (or more) images.")
            continue
        
        print(f"\nAugmenting for {person_name}. Generating {num_to_generate} new images...")
        
        generated_count = 0
        pbar = tqdm(total=num_to_generate, desc=f"  -> {person_name}")
        while generated_count < num_to_generate:
            random_image_name = random.choice(original_images)
            image_path = os.path.join(person_dir, random_image_name)
            
            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            augmented = transform(image=image)
            augmented_image = augmented['image']
            
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            
            # Save the new image in the same directory with a new name
            new_image_name = f"{person_name}_augmented_{num_originals + generated_count}.png"
            cv2.imwrite(os.path.join(person_dir, new_image_name), augmented_image)
            
            generated_count += 1
            pbar.update(1)
        pbar.close()
            
    print("\n✅ Data augmentation complete!")
    print(f"New images have been added to the folders inside: {DATA_PATH}")

if __name__ == "__main__":
    # Set up command-line argument parsing to accept a list of people
    parser = argparse.ArgumentParser(description="Augment face recognition dataset.")
    parser.add_argument(
        '--people', 
        nargs='*', # This allows for zero or more arguments
        help="A list of person folder names to augment. If not provided, all folders will be augmented."
    )
    args = parser.parse_args()
    
    augment_dataset(people_to_augment=args.people)