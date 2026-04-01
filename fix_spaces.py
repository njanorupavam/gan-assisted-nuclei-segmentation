import os

mask_dir = "data/masks"

for file in os.listdir(mask_dir):

    if " .png" in file:
        new_name = file.replace(" .png", ".png")

        old_path = os.path.join(mask_dir, file)
        new_path = os.path.join(mask_dir, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file} → {new_name}")

print("✅ All spaces removed")