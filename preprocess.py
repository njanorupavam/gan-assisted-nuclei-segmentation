import cv2

def preprocess_image(path):
    img = cv2.imread(path)

    # Resize
    img = cv2.resize(img, (256, 256))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize
    gray = gray / 255.0

    return img, gray