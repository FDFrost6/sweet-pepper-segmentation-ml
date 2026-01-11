import numpy as np
import time
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.color import rgb2lab
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.cluster import KMeans
"""
Task 1
Functions I used for Task 1, that I reused in the different solutions
"""

def timer(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        how_long = end - start
        print(f"It took {how_long:.4f}s")
        return result
    return wrapper

def eval_and_plot(name, segmenter, X, y):
    y_pred = segmenter.predicting(X)
    acc = accuracy_score(y, y_pred)
    cm  = confusion_matrix(y, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}\n")

    probs = segmenter.predict_proba(X)
    P, R, T = precision_recall_curve(y, probs)
    F1 = 2 * P * R / (P + R + 1e-8)
    best = np.argmax(F1)
    print(f"{name} Best F1={F1[best]:.4f} threshhold={T[best]:.2f}\n")

    plt.figure()
    plt.plot(R, P, label=name)
    plt.scatter(R[best], P[best], marker='X', label="Best")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR-Curve - {name}")
    plt.legend()
    plt.show()

@timer
def segment_image(segmenter, image):
    H, W, _ = image.shape
    pixels = image.reshape(-1,3)
    mask_flat = segmenter.predicting(pixels)
    return mask_flat.reshape(H, W)

"""
Task 1 - Solution1: Classifier Segmentation-class
"""

# Model from scratch
class LogisticRegression:

    def __init__(self, learningrate=0.001, max_iteration=100):

        self.weights = None
        self.biases = None
        self.learningrate = learningrate
        self.max_iteration = max_iteration

    def sigmoid(self, X):

        return 1/(1 + np.exp(-X))

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)
        self.biases = 0

        N0 = np.sum(y == 0)
        N1 = np.sum(y == 1)

        w0 = samples / (2 * N0)
        w1 = samples / (2 * N1)

        weight_per_sample = np.where(y ==1,w1,w0)
        for _ in range(self.max_iteration):
            preds = X.dot(self.weights) + self.biases

            y_preds = self.sigmoid(preds)

            error = (y_preds - y) * weight_per_sample
            dw = (1 / samples) * X.T.dot(error)
            db = (1 / samples) * np.sum(error)

            self.weights -= self.learningrate * dw
            self.biases -= self.learningrate * db

    def predict_proba(self, X):

        linear_model = X.dot(self.weights) + self.biases
        k1 = self.sigmoid(linear_model)
        k2 = 1 - k1
        return np.vstack([k2,k1]).T

    def predict(self, X):

        probs = self.predict_proba(X)[:,1]
        labels = (probs > 0.5).astype(int)
        return labels

"""
Task 1 - Solution1: Segmentation-class
"""
class PepperSegment:
    def __init__(self, model, color_space="rgb", threshhold=0.5):
        self.model       = model
        self.color_space = color_space
        self.threshhold  = threshhold
        # Scaler only for RGB
        self.scaler = StandardScaler() if color_space == "rgb" else None

    def preprocessing(self, X):
        # 1) Normiere RGB auf [0,1]
        Xrgb = X.astype(np.float32) / 255.0

        if self.color_space == "lab":
            # 2) RGB->Lab
            lab = rgb2lab(X)
            return lab.reshape(-1,3)

        return (Xrgb[:, 0] - Xrgb[:, 1]).reshape(-1, 1)

    def fitting(self, X, y):
        feats = self.preprocessing(X)
        if self.scaler is not None:
            feats = self.scaler.fit_transform(feats)
        self.model.fit(feats, y)

    def predict_proba(self, X):
        feats = self.preprocessing(X)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        return self.model.predict_proba(feats)[:,1]

    def predicting(self, X):
        feats = self.preprocessing(X)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats)[:,1]
        return (probs >= self.threshhold).astype(int)

"""
Task 1 - Solution2: Segmentation class 
"""

# Segmenter
class PepperSegment_SVM:
    def __init__(self, model, color_space="rgb", threshhold=0.5):

        self.model       = model
        self.color_space = color_space
        self.threshhold  = threshhold
        self.scaler = StandardScaler()

    def preprocessing(self, X):
        Xrgb = X.astype(np.float32) / 255.0

        if self.color_space == "lab":
            # jedes Pixel einzeln zu LAB wandeln, dann a & b
            lab_px = rgb2lab(X)  # (N,1,3)
            return lab_px.reshape(-1,3)

        return (Xrgb[:, 0] - Xrgb[:, 1]).reshape(-1, 1)

    def fitting(self, X, y):
        feats = self.preprocessing(X)
        feats = self.scaler.fit_transform(feats)
        self.model.fit(feats, y)

    def predicting(self, X):
        feats = self.preprocessing(X)
        feats = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats)[:,1]
        return (probs >= self.threshhold).astype(int)

    def predict_proba(self, X):
        feats = self.preprocessing(X)
        feats = self.scaler.transform(feats)
        return self.model.predict_proba(feats)[:,1]


"""
Task 2
Functions I used for Task 2, that I reused in the different solutions 
"""

def eval_plot_2(name, classifier, X, y):
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}\n")

    probs = classifier.predict_proba(X)[:, 1]

    P, R, T = precision_recall_curve(y, probs)
    F1 = 2 * P * R / (P + R + 1e-8)
    best = np.argmax(F1)
    print(f"{name} Best F1={F1[best]:.4f} threshhold={T[best]:.2f}\n")

    plt.figure()
    plt.plot(R, P, label=name)
    plt.scatter(R[best], P[best], marker='X', label="Best")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR-Curve - {name}")
    plt.legend()
    plt.show()

"""
Task 2 - Solution 1: Descriptor
"""

class BoVW:

    def __init__(self, n_clusters=30, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.kmeans     = None

    def fit(self, descriptors):
        """
        descriptors: array of shape (N_total, descriptor_dim)
        """
        self.kmeans = KMeans(n_clusters = self.n_clusters, max_iter=self.max_iter)
        self.kmeans.fit(descriptors)

    def transform(self, descriptors) :
        """
        Given descriptors for one image, assign each to nearest cluster
        and return normalized histogram of length n_clusters.
        """
        words = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(self.n_clusters+1))
        if hist.sum() > 0:
            hist = hist.astype(float) / hist.sum()
        return hist

"""
Task 2 - Solution 1: Classifier 
"""

class KNNClassifier:
    def __init__(self, k=5):
        self.k      = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self.X_train[None, :, :],axis=2)
        # Indizes der k-kleinsten Distanzen per Zeile
        nn = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        # Labels der Nachbarn
        votes = self.y_train[nn]  # (M, k)
        # Modus pro Zeile
        preds = np.array([np.bincount(v.astype(int)).argmax() for v in votes])
        return preds

    def predict_proba(self, X):
        dists = np.linalg.norm(X[:, None, :] - self.X_train[None, :, :],axis=2)
        nn = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        votes = self.y_train[nn]
        # Anteil positives Label pro Testpunkt
        probs_pos = np.mean(votes, axis=1)
        probs_neg = 1 - probs_pos
        # 2-Spalten-Matrix
        return np.vstack([probs_neg, probs_pos]).T

"""
Task 2 - Solution 1: Feature Exractor
"""

class HogExtractor:
    def __init__(self, orientations=9,ppc=(8,8), cpb=(2,2)):
        self.orientations = orientations
        self.ppc = ppc
        self.cpb = cpb

    def extract_feats(self, image):
        gray = rgb2gray(image)
        feat, _ = hog(gray,orientations=self.orientations,pixels_per_cell=self.ppc, cells_per_block=self.cpb, visualize=True,
            feature_vector=False)

        n_blocks = feat.size // self.orientations
        return feat.reshape((n_blocks, self.orientations))

"""
Task 2 - Solution 1: Functions to compute the Model 
"""

def compute_hog(hog_map, bovw, X):
    hists = []
    for img in X:
        desc = hog_map.extract_feats(img)
        hists.append(bovw.transform(desc))
    hists = np.vstack(hists)
    return hists

def bovw_builder(X_train, orientations, ppc, cpb, n_clusters):
    hog_map = HogExtractor(orientations=orientations,ppc=ppc,cpb=cpb)
    all_descriptors = []
    for img in X_train:
        desc = hog_map.extract_feats(img)
        all_descriptors.append(desc)
    all_descriptors = np.vstack(all_descriptors)

    #training
    bovw = BoVW(n_clusters=int(n_clusters))
    bovw.fit(all_descriptors)

    return hog_map, bovw





