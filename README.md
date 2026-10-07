# 🎾 Sports Equipment Classification using K-Nearest Neighbors (KNN)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-KNN%20Classifier-orange.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning project that classifies **14 types of sports balls and equipment** from images using **strictly the K-Nearest Neighbors (KNN)** algorithm. 

Unlike black-box deep learning models, this project demonstrates an engineered classical computer vision pipeline combining **color histograms, Histogram of Oriented Gradients (HOG), surface texture analysis, PCA dimensionality reduction, and hyperparameter-tuned KNN**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Supported Sports Equipment (14 Classes)](#-supported-sports-equipment-14-classes)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Feature Engineering Details](#-feature-engineering-details)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Dataset Preparation](#-dataset-preparation)
- [How to Train](#-how-to-train)
- [How to Predict / Test](#-how-to-predict--test)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🎯 Overview

| Component | Description |
|---|---|
| **Problem Domain** | Multi-Class Computer Vision Classification |
| **Algorithm** | **K-Nearest Neighbors (KNN)** (`sklearn.neighbors.KNeighborsClassifier`) |
| **Key Features** | Dual-Region HSV Color Histograms, HOG Shape Features, Laplacian & Canny Texture Descriptors |
| **Optimization** | Principal Component Analysis (PCA, 128 components) + 3-Fold Cross-Validation Grid Search |
| **Input Format** | Any standard image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) |
| **Output** | Predicted equipment class + Top-3 confidence probabilities + Matplotlib visual dialog |
| **Pretrained Model** | Provided as `knn_model.joblib` (~6.8 MB) for instant inference |

---

## 🏉 Supported Sports Equipment (14 Classes)

The model is trained to recognize 14 distinct categories:

| Index | Category | Index | Category |
|:---:|:---|:---:|:---|
| 1 | **American Football** | 8 | **Golf Ball** |
| 2 | **Baseball** | 9 | **Hockey Ball** |
| 3 | **Basketball** | 10 | **Rugby Ball** |
| 4 | **Billiards Ball** | 11 | **Shuttlecock** |
| 5 | **Bowling Ball** | 12 | **Table Tennis (Ping Pong) Ball** |
| 6 | **Cricket Ball** | 13 | **Tennis Ball** |
| 7 | **Football (Soccer)** | 14 | **Volleyball** |

---

## ⚙️ System Architecture & Pipeline

```mermaid
flowchart TD
    A[Input Image] --> B[Center Square Crop 1:1 Aspect Ratio]
    B --> C[Resize to 128x128 & Gaussian Blur 3x3]
    
    subgraph Feature Extraction
        C --> D1[Global HSV Histogram<br/>8x8x8 = 512 bins]
        C --> D2[Center Region HSV Histogram<br/>Inner 50% box x2.0 weight]
        C --> D3[HOG Shape & Edges<br/>Orientations: 9, Cell: 16x16]
        C --> D4[Texture & Seams<br/>Laplacian Variance + Canny Density]
        D1 & D2 & D3 & D4 --> E[Concatenated Feature Vector]
    end

    E --> F[StandardScaler Normalization]
    F --> G[PCA Dimensionality Reduction<br/>128 Components]
    G --> H[K-Nearest Neighbors Classifier<br/>GridSearch CV Tuned]
    H --> I[Top-3 Class Probabilities & Matplotlib Visualizer]
```

---

## 🔬 Feature Engineering Details

KNN relies directly on geometric distance in feature space. To give the model the discriminative power of human perception, four complementary feature representations are extracted:

1. **Aspect-Ratio Preserving Preprocessing:**
   - Instead of stretching the image, a center square crop is taken before resizing to `128x128` pixels. This preserves the genuine circular or prolate spheroidal shape of sports balls.
   - Light Gaussian blur ($3 \times 3$ kernel) removes high-frequency camera noise and compression artifacts.

2. **Global HSV Color Histogram (512 bins):**
   - Discretized into $8 \times 8 \times 8$ bins across Hue, Saturation, and Value channels.
   - Captures color signatures (e.g., bright yellow-green for tennis balls, deep red for cricket balls, vibrant orange for basketballs).

3. **Center-Region Focused HSV Color Histogram (Weighted $2.0\times$):**
   - The ball typically sits in the central $50\%$ region (`32:96, 32:96`).
   - Prioritizing the center prevents green grass, clay courts, or wooden gym floors from dominating the distance metric.

4. **HOG (Histogram of Oriented Gradients):**
   - Extracted using 9 gradient orientations, $16 \times 16$ pixels per cell, and $2 \times 2$ cells per block (`L2-Hys` block normalization).
   - Accurately captures structural geometry: the unique panel lines of a soccer ball, seams of a baseball, ribs of a basketball, or the skirt of a shuttlecock.

5. **Surface Texture & Edge Densities:**
   - **Laplacian Variance:** Distinguishes dimpled surfaces (golf balls) and rough seams from smooth gloss surfaces (billiard and bowling balls).
   - **Canny Edge Density:** Measures boundary transition sharpness.

6. **Dimensionality Reduction & Normalization:**
   - Features are standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
   - `PCA` projects the combined feature space down to 128 principal components, discarding redundant noise and drastically accelerating distance queries.

---

## 📁 Repository Structure

```
MachinneLearningModel/
├── training.py                 # Feature extraction, dataset loader, GridSearchCV, KNN training
├── prediction.py               # Interactive GUI file-picker & CLI prediction with top-3 confidence
├── knn_model.joblib            # Pre-trained KNN model pipeline (~6.8 MB) ready for inference
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git rules (excludes dataset & cache files)
├── AmericanFootball.jpeg       # Sample test image
├── CricketBall.jpeg            # Sample test image
├── Football.jpg                # Sample test image
├── GolfBall.jpg                # Sample test image
└── TennnisBall.jpeg            # Sample test image
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hunkar045/Sports-Equipment-KNN-Classification.git
cd Sports-Equipment-KNN-Classification
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy scikit-learn scikit-image opencv-python joblib matplotlib tqdm kagglehub
```

---

## 📊 Dataset Preparation

The model is built around the **Ball Classification Dataset 12k** (~12,700 images across 14 categories).

### Option A: Automatic Download (Default)
If the dataset is not detected locally, `training.py` will automatically download it via `kagglehub` from Kaggle:
```
Kaggle ID: abdullahpogla/ball-classification-dataset-12k
```

### Option B: Manual Placement
Download from [Kaggle](https://www.kaggle.com/datasets/abdullahpogla/ball-classification-dataset-12k), extract, and place in either:
- `./Ball Classification Integrated Dataset/` or
- `./dataset/`

```
Ball Classification Integrated Dataset/
├── train/
│   ├── american_football/
│   ├── baseball/
│   ├── basketball/
│   └── ... (14 class folders)
└── test/
    └── ... (14 class folders)
```

---

## 🏋️ How to Train

To train the model from scratch and tune hyperparameters:

```bash
python training.py
```

### What happens during training:
1. Multiprocessing pre-processes and extracts feature vectors across all images with `joblib.Parallel`.
2. A stratified 80% train / 20% test split is created (`random_state=42`).
3. An automated `GridSearchCV` evaluates 16 parameter combinations using 3-fold cross validation:
   - `n_neighbors`: `[3, 5, 7, 9]`
   - `weights`: `['uniform', 'distance']`
   - `metric`: `['euclidean', 'manhattan']`
4. The best estimator is evaluated on the held-out test set, printing test accuracy and a full classification report (precision, recall, F1-score).
5. The fitted pipeline (`StandardScaler` + `PCA` + `KNN`) is saved to `knn_model.joblib`.

> 💡 **Quick Test Run:** If you want a quick trial without waiting for the full dataset, edit line 38 in `training.py`:
> ```python
> MAX_PER_CLASS = 300   # Limits images per class for rapid testing
> ```

---

## 🔮 How to Predict / Test

You can test immediately using the pre-trained model `knn_model.joblib` without re-training!

### Method 1: Interactive GUI File Picker (Default)

Simply run:
```bash
python prediction.py
```
A file-selection dialog will pop up. Pick any image from your computer or choose one of the sample images provided in the repository (`Football.jpg`, `TennnisBall.jpeg`, etc.).

### Method 2: Command Line Argument

Provide the path to any image directly:
```bash
python prediction.py Football.jpg
```
or
```bash
python prediction.py AmericanFootball.jpeg
```

### Sample Terminal Output:

```text
Prediction: TENNIS
  tennis                    89.4%
  table_tennis_ball          6.2%
  hockey_ball                4.4%
```

A visual window pops up displaying the original test image alongside its predicted class and confidence score.

---

## 📈 Model Evaluation & Results

- **Dimensionality Reduction:** Compresses thousands of raw pixel and gradient dimensions into 128 PCA components, capturing maximum variance while retaining real-time prediction speed.
- **Distance Weighting:** Distance-weighted neighbors (`weights='distance'`) give closer matches significantly more influence, outperforming naive uniform voting.
- **Separation of Ball from Background:** Center-weighting HSV color prevents common misclassifications caused by green grass (e.g., differentiating football/cricket from tennis).

---

## ❓ Troubleshooting & FAQ

| Issue | Solution |
|---|---|
| `Model not found. Run python training.py first.` | Ensure `knn_model.joblib` is present in the workspace root or execute `python training.py`. |
| `Tkinter window does not appear` | If running on headless Linux, install `sudo apt install python3-tk` or pass the image path directly via CLI: `python prediction.py <image_path>`. |
| `Memory error during training` | Reduce `PCA_COMPONENTS` or set `MAX_PER_CLASS = 500` in `training.py`. |
| `Kaggle download prompt` | If automatic download is triggered, ensure Kaggle API credentials (`kaggle.json`) are configured in `~/.kaggle/` or download manually. |

---

## 👤 Author

- **Hunkar Chaware** ([@hunkar045](https://github.com/hunkar045))
- GitHub: [hunkar045](https://github.com/hunkar045)
- College Machine Learning Model Project

---

## 📜 Acknowledgments

- Dataset by **abdullahpogla** on Kaggle: [Ball Classification Dataset 12k](https://www.kaggle.com/datasets/abdullahpogla/ball-classification-dataset-12k).
- Scikit-learn and OpenCV open-source communities.
