from preprocess import preprocess_image
from detect import detect_nuclei
import cv2

# Load image
img_path = "data/images/sample.png"

original, gray = preprocess_image(img_path)

# Detect nuclei
output, thresh = detect_nuclei(original, gray)

# Save outputs
cv2.imwrite("output/detected.png", output)
cv2.imwrite("output/thresh.png", thresh)

print("✅ Detection completed. Check output folder.")