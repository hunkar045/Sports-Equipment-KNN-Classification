"""
Sports ball classification - TRAINING (KNN only)

Usage:
    python training.py

Put the Kaggle dataset inside a folder named  dataset/  or  Ball Classification Integrated Dataset/
next to this file. If neither is found, the script tries to download it automatically with kagglehub.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier      # the ONLY classifier used
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------- settings ----------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Ball Classification Integrated Dataset"
if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "knn_model.joblib"
KAGGLE_ID = "abdullahpogla/ball-classification-dataset-12k"

IMG_SIZE = (128, 128)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_SIZE = 0.2
RANDOM_STATE = 42
PCA_COMPONENTS = 128
MAX_PER_CLASS = None        # set e.g. 300 for a quick test run


# ---------------- preprocessing (also used by prediction.py) ----------------
def preprocess_image(path):
    """Read -> square center-crop/resize to 128x128 -> light blur. Preserves ball circularity."""
    if isinstance(path, (str, Path)):
        data = np.fromfile(str(path), np.uint8)
        if len(data) == 0:
            raise ValueError(f"Empty image file: {path}")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)           # always 3-channel
    else:
        img = path
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    # Center square crop to preserve 1:1 aspect ratio so circular balls aren't distorted
    h, w = img.shape[:2]
    min_dim = min(h, w)
    sy = (h - min_dim) // 2
    sx = (w - min_dim) // 2
    square = img[sy:sy + min_dim, sx:sx + min_dim]

    img_resized = cv2.resize(square, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img_resized, (3, 3), 0)


def extract_features(path):
    """
    HSV colour histogram (global + center ball region) + HOG shape features + texture.
    Separating center ball color from the background prevents grass/court colors from misleading KNN.
    """
    img = preprocess_image(path)

    # 1. Global HSV colour histogram (8x8x8 = 512 bins)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_glob = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                             [0, 180, 0, 256, 0, 256])
    hist_glob = cv2.normalize(hist_glob, hist_glob).flatten()

    # 2. Center region HSV histogram (inner 50% box where the ball is located)
    c_box = hsv[32:96, 32:96]
    hist_center = cv2.calcHist([c_box], [0, 1, 2], None, [8, 8, 8],
                               [0, 180, 0, 256, 0, 256])
    hist_center = cv2.normalize(hist_center, hist_center).flatten()

    # 3. Shape / edge features (HOG)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), block_norm="L2-Hys")

    # 4. Center texture & edge features (dimpled/seamed balls vs smooth balls)
    gray_c = gray[32:96, 32:96]
    lap_var = np.array([cv2.Laplacian(gray_c, cv2.CV_64F).var()], dtype=np.float32)
    canny_dens = np.array([np.mean(cv2.Canny(gray_c, 50, 150) > 0)], dtype=np.float32)

    return np.concatenate([hist_glob, hist_center * 2.0, shape, lap_var * 0.005, canny_dens]).astype(np.float32)


# ---------------- dataset loading ----------------
def find_images(root):
    """Label = name of the folder that directly contains the image."""
    by_class = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.parent != Path(root):
            by_class.setdefault(p.parent.name.strip().lower(), []).append(p)
    return by_class


def load_dataset():
    by_class = find_images(DATASET_DIR) if DATASET_DIR.exists() else {}
    if not by_class:
        print("Dataset not found locally, downloading from Kaggle...")
        import kagglehub
        by_class = find_images(kagglehub.dataset_download(KAGGLE_ID))
    if not by_class:
        raise SystemExit("No images found. Put the dataset in the 'dataset' folder.")

    rng = np.random.RandomState(RANDOM_STATE)
    paths, labels = [], []
    for cls, files in by_class.items():
        if MAX_PER_CLASS and len(files) > MAX_PER_CLASS:
            files = [files[i] for i in rng.choice(len(files), MAX_PER_CLASS, replace=False)]
        paths += files
        labels += [cls] * len(files)

    def safe(p):
        try:
            return extract_features(p)
        except Exception:
            return None

    feats = Parallel(n_jobs=-1)(delayed(safe)(p) for p in tqdm(paths, desc="Preprocessing"))
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.vstack([feats[i] for i in keep])
    y = np.array(labels)[keep]
    print(f"Loaded {len(y)} images from {len(set(y))} classes")
    return X, y


# ---------------- training ----------------
def main():
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    n_comp = min(PCA_COMPONENTS, int(len(X_tr) * 0.6) - 1, X_tr.shape[1])
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier()),
    ])
    grid = {
        "knn__n_neighbors": [3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    search = GridSearchCV(pipe, grid, cv=3, n_jobs=-1, verbose=1)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_
    print("Best KNN settings:", search.best_params_)

    pred = model.predict(X_te)
    print(f"\nTest accuracy: {accuracy_score(y_te, pred):.4f}\n")
    print(classification_report(y_te, pred))

    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
