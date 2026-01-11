from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
# =========================
# Evaluation Utilities
# =========================
def print_knn_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}")
    print(f"{name} Precision: {precision:.3f}, Recall: {recall:.3f}")
    print()

# =========================
# Imports
# =========================
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from collections import Counter
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report

# =========================
# Color Space & Normalization
# =========================
def convert_color_space(X, color_space):
    """
    Convert features to the specified color space (rgb or lab).
    Assumes input X is (n_samples, n_features) and features are RGB if n_features==3.
    """
    X = np.array(X, dtype=np.float32)
    if color_space == 'lab':
        if X.shape[1] == 3:
            # Normalize RGB to [0,1] before Lab conversion
            X_rgb = X / 255.0
            X_rgb = X_rgb.reshape(-1, 1, 3)
            X_lab = color.rgb2lab(X_rgb)
            return X_lab.reshape(-1, 3)
        else:
            return X
    else:
        return X

def manual_normalize(X):
    X = np.array(X, dtype=np.float32)
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    return (X - min_vals) / denom

# =========================
# KNN Utilities
# =========================



# Simple, non-batched, non-parallel KNN
class CustomKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)

    def predict(self, X):
        from scipy.stats import mode
        X = np.array(X, dtype=np.float32)
        cls = np.zeros((X.shape[0],))
        for i, x in enumerate(X):
            dists = np.linalg.norm(self.X_train - x, axis=1)
            ids = dists.argsort()[:self.n_neighbors]
            nn = self.y_train[ids]
            cls[i], _ = mode(nn)
        return cls

def train_custom_knn(X_train, y_train, n_neighbors=5):
    knn = CustomKNNClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn



# =========================
# Segmentation Utility
# =========================
def segment_image_with_knn(image_path, knn, color_space='rgb', batch_size=None, save_path=None, class_colors=None):
    from skimage.io import imread, imsave
    image = imread(image_path)
    if image.max() > 1.0:
        image = image / 255.0
    orig_shape = image.shape[:2]
    pixels = image.reshape(-1, 3)
    pixels = manual_normalize(convert_color_space(pixels, color_space))
    if hasattr(knn, 'predict') and 'batch_size' in knn.predict.__code__.co_varnames:
        predictions = knn.predict(pixels, batch_size=batch_size)
    else:
        predictions = knn.predict(pixels)
    segmentation = predictions.reshape(orig_shape)
    if class_colors is None:
        plt.imshow(segmentation, cmap='jet')
        plt.title('KNN Segmentation')
        if save_path:
            plt.savefig(save_path)
        plt.show()
    else:
        mask_rgb = np.zeros((orig_shape[0], orig_shape[1], 3), dtype=np.uint8)
        for label, color in class_colors.items():
            mask_rgb[segmentation == label] = color
        plt.imshow(mask_rgb)
        plt.title('KNN Segmentation')
        if save_path:
            imsave(save_path, mask_rgb)
        plt.show()
    return segmentation

# =========================
# Plotting & Reporting
# =========================
def plot_knn_data_distribution(X, y, title="Data Distribution"):
    y = np.array(y)
    X_2d = np.array(X)[:, :2]
    classes = np.unique(y)
    for cls in classes:
        plt.scatter(X_2d[y == cls, 0], X_2d[y == cls, 1], label=f"Class {cls}", alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_precision_recall_standard(y_true, y_proba, class_names=None, title="Precision-Recall Curve"):
    if len(y_proba.shape) == 1 or y_proba.shape[1] == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP={ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.show()
    else:
        n_classes = y_proba.shape[1]
        if class_names is None:
            class_names = [str(i) for i in range(n_classes)]
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
            ap = average_precision_score(y_true == i, y_proba[:, i])
            plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend()
        plt.show()

def print_classification_report(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names))