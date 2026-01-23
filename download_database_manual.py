import os
import shutil

def extract_images(source_folder, destination_folder):
    # 1. Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    count = 0
    
    # 2. Walk through the source directory (recursive search)
    print("Starting extraction... this may take a moment.")
    
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check for .jpg extension (case insensitive)
            if file.lower().endswith(('.jpg', '.jpeg')):
                
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)

                # 3. Handle Duplicate Filenames
                # If file exists in destination, rename the new one (e.g., photo.jpg -> photo_1.jpg)
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(destination_path):
                    new_filename = f"{base}_{counter}{extension}"
                    destination_path = os.path.join(destination_folder, new_filename)
                    counter += 1

                # 4. Copy the file
                # copy2 preserves metadata (timestamps, etc.)
                try:
                    shutil.copy2(source_path, destination_path)
                    count += 1
                    
                    # Optional: Print progress every 500 images
                    if count % 500 == 0:
                        print(f"Processed {count} images...")
                        
                except Exception as e:
                    print(f"Error copying {source_path}: {e}")

    print(f"--- Completed ---")
    print(f"Total images extracted: {count}")

# --- CONFIGURATION ---
# Replace these paths with your actual folder paths
# Windows Example: r"C:\Users\Name\Downloads\SourceData"
# Mac/Linux Example: "/Users/Name/Downloads/SourceData"

src_dir = r"C:\Users\himan\Downloads\imga\lfw-funneled\lfw_funneled"
dest_dir = r"<REST OF THE PATH>\computer-vision-security-sys\data\raw\background"

# Run the function
if __name__ == "__main__":
    extract_images(src_dir, dest_dir)