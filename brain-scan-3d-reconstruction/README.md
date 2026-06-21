# Brain Scan 3D Reconstruction

This project implements a complete pipeline for **3D reconstruction from 2D brain scan slices** using classical computer vision techniques. It is designed as an educational project to learn image registration, feature detection, geometric transformations, and volume reconstruction.

---

## Project Overview

The goal is to take a stack of unregistered 2D brain slice images (e.g., from microscopy or histology) and align them into a coherent 3D volume. This involves:

1. **Image Registration** — Estimating geometric transformations (rigid / affine) between adjacent slices.
2. **Feature Detection & Matching** — Automatically finding corresponding points using SIFT and robust matching with RANSAC.
3. **Image Filtering & Preprocessing** — Downsampling, histogram equalization, and filtering (Gaussian, median, mean, Sobel).
4. **3D Volume Assembly** — Concatenating transformations and stacking aligned slices into a 3D volume for visualization.

---

## Repository Structure

| File | Description |
|------|-------------|
| `brain_scan_3d_reconstruction.ipynb` | Main Jupyter notebook with guided exercises (1–4) leading to the full 3D reconstruction pipeline. |
| `image_registration_functions.py` | Student implementation file containing helper functions for transformations, filtering, feature extraction, matching, and RANSAC. |
| `requirements.txt` | Python dependencies required to run the project. |
| `README.md` | This file. |

---

## Exercises Breakdown

### Exercise 1 — Manual Image Registration
- Load and visualize TIFF brain scan images.
- Manually annotate corresponding keypoints between adjacent slices.
- Estimate **affine transformations** from point correspondences.
- Apply backward mapping to align and overlay images.

**Key concepts:** homogeneous coordinates, least-squares estimation, backward mapping.

### Exercise 2 — Image Filtering & Preprocessing
- **Downsampling** using bilinear interpolation.
- **Histogram equalization** for intensity normalization.
- Implement and compare filters: **mean**, **median**, **Gaussian**, and **Sobel edge detection**.
- Extend downsampling with **anti-aliasing** via Gaussian pre-filtering.

**Key concepts:** interpolation, convolution, anti-aliasing, contrast enhancement.

### Exercise 3 — Automatic Feature Matching
- Detect keypoints and descriptors using **SIFT** (via OpenCV or scikit-image).
- Match descriptors using **distance-ratio testing** and **cross-checking**.
- Remove outliers with **RANSAC** to estimate robust affine transformations.
- Visualize matches and alignment results.

**Key concepts:** feature descriptors, nearest-neighbor matching, robust model fitting.

### Exercise 4 — Full 3D Reconstruction
- Compute pairwise transformations between all adjacent slices.
- Concatenate transformations to align the entire stack to a reference slice.
- Stack aligned images into a **3D volume**.
- Inspect the final reconstruction interactively with **napari**.

**Key concepts:** transformation chaining, error propagation, volume visualization.

---

## Dependencies

Install required packages with:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy` — numerical operations
- `scipy` — scientific computing (distance metrics, filtering)
- `matplotlib` — plotting and visualization
- `opencv-python` — image I/O and feature detection (SIFT)
- `scikit-image` — image processing and geometric transforms
- `tifffile` — reading TIFF microscopy images
- `Pillow` — image loading utilities
- `napari` — interactive 3D volume visualization
- `jupyter` — notebook environment

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the notebook:**
   ```bash
   jupyter notebook brain_scan_3d_reconstruction.ipynb
   ```

3. **Follow the exercises** in the notebook. Implement the missing functions in `image_registration_functions.py` as indicated by `TODO` comments.

4. **Data:** The notebook expects brain scan images under `data/tiff/` and `data/png/`, along with optional pre-annotated keypoints under `data/keypoints_ex1/`.

---

## Notes

- The `image_registration_functions.py` file is intentionally left with empty `TODO` stubs for educational purposes. Fill them in as you progress through the exercises.
- The notebook includes timing benchmarks and visualization helpers to verify your implementations at each step.
- For the best experience, use `napari` to interactively explore the final 3D reconstructed brain volume.

---

## Learning Outcomes

By completing this project, you will gain hands-on experience with:
- Classical image registration pipelines
- Geometric transformations and backward mapping
- Feature detection, description, and matching
- Robust estimation with RANSAC
- Building 3D volumes from 2D slices
- Practical use of OpenCV, scikit-image, and napari

---

*Project for educational use — Image Analysis & Computer Vision course.*
