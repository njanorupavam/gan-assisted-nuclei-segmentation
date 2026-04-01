import os
import cv2
import numpy as np
from model import build_unet
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 256

# 🔥 PASTE THE FUNCTION HERE 👇
def load_data(img_dir, mask_dir):
    images = []
    masks = []

    for file in os.listdir(img_dir):

        if not (file.endswith(".png") or file.endswith(".tif")):
            continue

        img_path = os.path.join(img_dir, file)

        # Convert .tif → .png for mask
        mask_name = file.replace(".tif", ".png")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"⚠️ Mask missing: {mask_name}")
            continue

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

        images.append(img)
        masks.append(mask)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    masks = np.array(masks).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return images, masks


# 🔽 LOAD DATA (already exists in your code)
img_dir = "data/images"
mask_dir = "data/masks"

# Check if directories exist and have data
if not os.path.exists(img_dir):
    os.makedirs(img_dir, exist_ok=True)
    print(f"❌ Error: {img_dir} directory is empty. Please add training images.")
    exit(1)

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    print(f"❌ Error: {mask_dir} directory is empty. Please add mask images.")
    exit(1)

if len(os.listdir(img_dir)) == 0:
    print(f"❌ Error: {img_dir} is empty. Please add training images.")
    exit(1)

if len(os.listdir(mask_dir)) == 0:
    print(f"❌ Error: {mask_dir} is empty. Please add mask images.")
    exit(1)

X, y = load_data(img_dir, mask_dir)

if len(X) == 0:
    print("❌ Error: No valid image-mask pairs found. Check your data directories.")
    exit(1)

print("Images:", len(X))
print("Masks:", len(y))


# 🔽 BUILD MODEL
model = build_unet()

model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 🔽 TRAIN
model.fit(X, y, epochs=50, batch_size=2)

# 🔽 SAVE MODEL
model.save("unet_model.h5")

print("✅ Model trained and saved!")