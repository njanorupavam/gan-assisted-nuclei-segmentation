# GAN-Assisted Nuclei Segmentation (Beginner-Friendly)

This repository contains a simple nuclei detection pipeline with a GAN preprocessor and U-Net segmentation model. 
It is intended for beginners (with or without coding experience) and shows a full workflow: data prep, training, and prediction.

## 📁 Project Structure

- `data/`
  - `images/`: raw images (input)
  - `masks/`: binary masks (labels) for training
- `output/`: generated predictions and overlays
- `archive/`: example original dataset path structure (optional)

- `preprocess.py`: image preprocessing utility
  - reads an image, resizes to 256×256, converts to grayscale, and normalizes pixel values.

- `model.py`: model architecture definition
  - contains `build_unet()` function to create a U-Net model for segmentation.

- `detect.py`: model-free detection prototype
  - uses threshold + contour logic for nucleus detection without deep learning.

- `main.py`: simple pipeline invocation (preprocess + detect) for demo
  - loads one image from `data/images/sample.png`, runs detection, and writes to `output/`.

- `train.py`: train the U-Net model with labeled data
  - loads `data/images/` + `data/masks/`, trains U-Net for 50 epochs, saves `unet_model.h5`.

- `train_gan.py`: GAN training placeholder file (if available)
  - trains `gan_generator.h5` or references GAN process.

- `predict.py`: prediction pipeline using GAN + U-Net
  - loads `gan_generator.h5` and `unet_model.h5`.
  - reads raw input image, (GAN enhancement), U-Net prediction, outputs:
    - `output/gan_binary_mask.png`
    - `output/gan_colored_mask.png`
    - `output/gan_overlay.png`

- `xml_to_mask.py`: converts image+XML annotations to binary masks
  - supports `.png` and `.tif` files, draws polygons into mask array.

## 🛠️ Prerequisites

Install Python dependencies (recommended inside virtual environment):

```powershell
python -m pip install --upgrade pip
pip install opencv-python numpy tensorflow
```

## 🚀 Step-by-step Usage

### 1. Prepare your training data

- `data/images/` should contain image files (`.png` or `.tif`).
- `data/masks/` should contain corresponding mask files (`.png`), same base name.

Example:
- `data/images/train1.png`
- `data/masks/train1.png`

### 2. Create masks from XML (optional)

If you have XML annotations, run:
```powershell
python xml_to_mask.py
```
This fills `data/images` and `data/masks` folders with 256×256 pairs.

### 3. Train U-Net model

```powershell
python train.py
```

- Output: `unet_model.h5`
- If folder is empty, script prints a friendly error and stops.

### 4. Train GAN (optional)

```powershell
python train_gan.py
```
- Output: `gan_generator.h5`

### 5. Run prediction

```powershell
python predict.py
```
- Uses both models.
- For a different input image, change `IMAGE_PATH` in `predict.py`.

### 6. Run the simple detector demo

```powershell
python main.py
```
- Produces `output/detected.png` and `output/thresh.png`.

## 🧩 How each file is connected

- `xml_to_mask.py` generates training labels (`data/masks`) from XML annotations.
- `train.py` reads those labels, trains a U-Net and saves `unet_model.h5`.
- `train_gan.py` (optional) trains GAN model and saves `gan_generator.h5`.
- `predict.py` loads both models and generates final mask outputs.
- `main.py` shows basic classic CV detection path (no model required).

## ✅ Final output explanation

After `python predict.py`, check `output/`:
- `gan_binary_mask.png`: binary segmentation mask (0/255).
- `gan_colored_mask.png`: heatmap from predicted probabilities.
- `gan_overlay.png`: overlay of segmentation on the original image.

## 🧹 GitHub Push (if not done already)

```powershell
cd c:\Users\kaasi\nucleidet\nuclei_detection
git remote add origin https://github.com/njanorupavam/gan-assisted-nuclei-segmentation.git
git branch -M main
git push -u origin main
```

> If models exceed file size limit (>100MB), use Git LFS:
> `git lfs install` and `git lfs track "*.h5"`.

## 📝 Notes for non-tech users

- You can run commands by copy-paste one line at a time.
- If any command says a folder is empty, add the required files there.
- No coding skill is needed beyond editing `IMAGE_PATH` or checking folder contents.

---

If you want, I can also add a short CLI wrapper `run.sh` / `run.bat` to run the full pipeline with one command.`
>>>>>>> d24c1e6 (Add beginner-friendly README)
