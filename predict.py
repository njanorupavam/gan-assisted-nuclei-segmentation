import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔹 SETTINGS
IMG_SIZE = 256
MODEL_PATH = "unet_model.h5"
IMAGE_PATH = "data/images/TCGA-DK-A2I6-01A-01-TS1.tif"  # change if needed

# 🔹 LOAD MODELS
model = load_model(MODEL_PATH)
gan = load_model("gan_generator.h5")

# 🔹 READ IMAGE
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"❌ Error: Cannot load image → {IMAGE_PATH}")
    exit()

original = img.copy()

# 🔹 RESIZE + NORMALIZE
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0

# Prepare input
input_img = np.reshape(img, (1, 256, 256, 1))

# 🔥 GAN enhancement
enhanced = gan.predict(input_img)[0]

# Use enhanced image for U-Net
input_img = np.reshape(enhanced, (1, 256, 256, 1))

# 🔹 PREDICTION
pred = model.predict(input_img)[0]

# 🔹 BINARY MASK (clean segmentation)
binary_mask = (pred > 0.5).astype("uint8") * 255

# 🔹 SMOOTH MASK (for visualization)
smooth_mask = (pred * 255).astype("uint8")

# 🔹 COLOR MAP OUTPUT
colored_mask = cv2.applyColorMap(smooth_mask, cv2.COLORMAP_JET)

# 🔹 RESIZE BACK TO ORIGINAL SIZE
binary_mask = cv2.resize(binary_mask, (original.shape[1], original.shape[0]))
colored_mask = cv2.resize(colored_mask, (original.shape[1], original.shape[0]))

# 🔹 CREATE OVERLAY (very useful)
overlay = cv2.addWeighted(original, 0.7, binary_mask, 0.3, 0)

# 🔹 CREATE OUTPUT FOLDER
os.makedirs("output", exist_ok=True)

# 🔹 SAVE RESULTS
cv2.imwrite("output/gan_binary_mask.png", binary_mask)
cv2.imwrite("output/gan_colored_mask.png", colored_mask)
cv2.imwrite("output/gan_overlay.png", overlay)

print("✅ Prediction complete!")
print("Saved:")
print(" - output/gan_binary_mask.png")
print(" - output/gan_colored_mask.png")
print(" - output/gan_overlay.png")