import numpy as np
from sklearn.ensemble import RandomForestClassifier
from libs.logreg_utils import plot_precision_recall_curve, timing

class MyPCA:
    """
    Custom PCA implementation using numpy (SVD).
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        # Center the data
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class PCA_RF_Classifier:
    """
    Object-oriented classifier using custom PCA for feature extraction and RandomForest for classification.
    """
    def __init__(self, n_components=20, n_estimators=100, random_state=42, class_weight=None):
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_weight = class_weight
        self.pca = MyPCA(n_components=self.n_components)
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight=self.class_weight
        )

    def fit(self, X, y):
        # Flatten images if needed (for image arrays)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        X_pca = self.pca.fit_transform(X)
        self.rf.fit(X_pca, y)

    def predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        X_pca = self.pca.transform(X)
        return self.rf.predict(X_pca)

    def predict_proba(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        X_pca = self.pca.transform(X)
        return self.rf.predict_proba(X_pca)
