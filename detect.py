import cv2
import numpy as np

def detect_nuclei(original, gray):
    gray = (gray * 255).astype("uint8")

    # Apply threshold
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours (nuclei-like regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = original.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 1)

    return output, thresh