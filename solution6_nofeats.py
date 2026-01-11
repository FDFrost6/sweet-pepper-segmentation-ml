from libs.data_utils import load_q2_data, rgb_to_hsv_flat
from libs.svm_utils import process_input_imageQ2_p2
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='path to input image')
args = parser.parse_args()

def flatten_hsv_images(X):
    """
    Converts a batch of RGB images to HSV and flattens to 2D array (num_samples, num_pixels * num_channels).
    Uses rgb_to_hsv_flat from libs.dataloader.
    """
    hsv_images = []
    for img in X:
        hsv = rgb_to_hsv_flat(img)
        hsv_images.append(hsv.flatten())
    return np.array(hsv_images)


def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()
    X_train_flat = flatten_hsv_images(X_train)
    X_valid_flat = flatten_hsv_images(X_valid)

    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    pca = PCA(n_components=20, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_valid_pca = pca.transform(X_valid_flat)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_pca, y_train, sample_weight=sample_weight)

    y_valid_pred = clf.predict(X_valid_pca)
    y_valid_proba = clf.predict_proba(X_valid_pca)[:, 1]
    print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_valid, y_valid_pred))
    print("Classification Report:\n", classification_report(y_valid, y_valid_pred, target_names=["Not Pepper", "Sweet Pepper"]))

    process_input_imageQ2_p2(args.input_image, clf, pca)

if __name__ == "__main__":
    main()
