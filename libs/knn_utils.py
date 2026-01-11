
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
from skimage import color
from sklearn.preprocessing import MinMaxScaler
import functools
from collections import Counter


############################
#TIMER AND COLORSPACE UTILS#
############################

#color space conversion
def convert_color_space(X, color_space):
    """
    Convert features to the specified color space (rgb or lab).
    Assumes input X is (n_samples, n_features) and features are RGB if n_features==3.
    """
    X = np.array(X)
    if color_space == 'lab':
        if X.shape[1] == 3:
            X_rgb = X.reshape(-1, 1, 3)
            X_lab = color.rgb2lab(X_rgb)
            return X_lab.reshape(-1, 3)
        else:
            #If more than 3 features, assume already processed or not color
            return X
    else:
        return X


def time_function(func):
    """
    Decorator to time any function call and print elapsed time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


###############################
# CUSTOM KNN CLASS & UTILITIES#
###############################

def calc_batch_size(ram_gb, n_train, n_features):
    """
    Calculate batch size for KNN batching given RAM in GB, number of training samples, and number of features.
    """
    memory_limit = ram_gb * 1024**3  # Convert GB to bytes
    batch_size = int(memory_limit // (n_train * n_features * 8))
    return max(1, batch_size)


class CustomKNNClassifier:
    """
    Custom implementation of KNN classifier (object-oriented).
    Implements fit and predict methods.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X, batch_size=None):
        """
        Batched prediction for memory efficiency and speed.
        If batch_size is None, defaults to safe value for 5GB RAM based on training set size and features.
        """
        X = np.array(X)
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        n_features = self.X_train.shape[1]
        # Calculate batch size for 5GB RAM
        if batch_size is None:
            # 5GB = 5,368,709,120 bytes
            memory_limit = 5_368_709_120
            batch_size = int(memory_limit // (n_train * n_features * 8))
            batch_size = max(1, batch_size)  # Ensure at least 1
        y_pred = []
        # Explanation for batching:
        # Each batch computes a distance matrix of shape (batch_size, n_train, n_features)
        # Each float64 value uses 8 bytes
        # For 5GB RAM: batch_size = 5,368,709,120 / (n_train * n_features * 8)
        # This batching allows efficient vectorized computation without exceeding memory limits
        for start in tqdm(range(0, n_test, batch_size), desc="CustomKNN Batched Predict", unit="batch"):
            end = min(start + batch_size, n_test)
            X_batch = X[start:end]
            # Vectorized distance computation for the batch
            distances = np.linalg.norm(self.X_train[None, :, :] - X_batch[:, None, :], axis=2)
            for i in range(distances.shape[0]):
                nn_indices = np.argsort(distances[i])[:self.n_neighbors]
                nn_labels = self.y_train[nn_indices]
                most_common = Counter(nn_labels).most_common(1)[0][0]
                y_pred.append(most_common)
        return np.array(y_pred)



def manual_normalize(X):
    """
    Manually normalize features to [0, 1] using min-max scaling (for custom KNN).
    """
    X = np.array(X)
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    return (X - min_vals) / denom

@time_function
def train_custom_knn(X_train, y_train, n_neighbors=5):
    """Train a CustomKNNClassifier and return the trained model."""
    knn = CustomKNNClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

@time_function
def predict_custom_knn(knn, X, desc="Predicting", batch_size=None):
    """Predict with CustomKNNClassifier (no proba). Returns y_pred. Supports batching."""
    with tqdm(total=1, desc=desc) as pbar:
        y_pred = knn.predict(X, batch_size=batch_size)
        pbar.update(1)
    return y_pred


#######################
#SKLEARN KNN UTILITIES#
#######################

def normalize_features(X):
    """
    Normalize features to [0, 1] using MinMaxScaler (for sklearn KNN).
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

@time_function
def train_knn(X_train, y_train, n_neighbors=5):
    """Train a KNN classifier and return the trained model."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# KNN prediction with progress bar (sklearn)
@time_function
def predict_with_progress(knn, X, desc="Predicting"):
    """Predict with tqdm progress bar."""
    with tqdm(total=1, desc=desc) as pbar:
        y_pred = knn.predict(X)
        y_proba = knn.predict_proba(X)
        pbar.update(1)
    return y_pred, y_proba

############################
#METRICS AND PLOT UTILITIES#
############################

# Data distribution plot
def plot_knn_data_distribution(X, y, title="Data Distribution"):
    """
    Plot the data distribution for KNN as scatter plot.
    """
    # Use only the first two features for 2D plot
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

# Standardized precision-recall curve for binary and multi-class
def plot_precision_recall_standard(y_true, y_proba, class_names=None, title="Precision-Recall Curve"):
    """
    Plot precision-recall curves for binary or multi-class classification.
    """
    # If y_proba is 1D, treat as binary
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

# Classification report
def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print classification report for KNN
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


