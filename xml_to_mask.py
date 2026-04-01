import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

IMG_SIZE = 256

# 🔹 Function to create mask from XML
def create_mask_from_xml(xml_path, image_path):
    original_img = cv2.imread(image_path)

    if original_img is None:
        print(f"❌ Cannot read image: {image_path}")
        return None

    h, w = original_img.shape[:2]

    # Empty mask
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for region in root.iter('Region'):
        points = []

        for vertex in region.iter('Vertex'):
            x = int(float(vertex.attrib['X']))
            y = int(float(vertex.attrib['Y']))
            points.append([x, y])

        if len(points) == 0:
            continue

        points = np.array(points, dtype=np.int32)

        # 🔹 Scale points to 256x256
        points[:, 0] = points[:, 0] * IMG_SIZE // w
        points[:, 1] = points[:, 1] * IMG_SIZE // h

        # Fill polygon
        cv2.fillPoly(mask, [points], 255)

    return mask


# 🔹 SET YOUR DATASET PATH HERE
image_dir = "archive/kmms_test/kmms_test/images"
xml_dir = "archive/kmms_test/kmms_test/annotations"

# 🔹 OUTPUT PATH (inside your project)
output_img_dir = "data/images"
output_mask_dir = "data/masks"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)


# 🔹 MAIN LOOP
for file in os.listdir(image_dir):

    if not (file.endswith(".png") or file.endswith(".tif")):
        continue

    print(f"🔄 Processing: {file}")

    img_path = os.path.join(image_dir, file)
    xml_path = os.path.join(xml_dir, file.replace(".png", ".xml"))

    if not os.path.exists(xml_path):
        print(f"⚠️ No XML found for: {file}")
        continue

    # Read and resize image
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Failed to load image: {file}")
        continue

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Create mask
    mask = create_mask_from_xml(xml_path, img_path)

    if mask is None:
        continue

    # Save outputs
    cv2.imwrite(os.path.join(output_img_dir, file), img_resized)
    cv2.imwrite(os.path.join(output_mask_dir, file), mask)


print("✅ Dataset ready! Masks generated successfully.")