from libs.data_utils import load_q2_data, rgb_to_hsv_flat
from libs.svm_utils import extract_hsv_histogram_skimage
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='path to input image')
args = parser.parse_args()

def extract_combined_features(X, h_bins=8, s_bins=4, v_bins=4):
    """
    Concatenates HSV histogram features and flattened HSV image features for a batch of images.
    """
    feats = []
    for img in X:
        hsv = rgb_to_hsv_flat(img)
        hsv_flat = hsv.flatten()
        hist = extract_hsv_histogram_skimage(img, h_bins=h_bins, s_bins=s_bins, v_bins=v_bins)
        combined = np.concatenate([hsv_flat, hist])
        feats.append(combined)
    return np.array(feats)

def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()
    X_train_combined = extract_combined_features(X_train)
    X_valid_combined = extract_combined_features(X_valid)

    scaler = StandardScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)
    X_valid_combined = scaler.transform(X_valid_combined)

    X_pepper = X_train_combined[y_train == 1]
    X_bg = X_train_combined[y_train == 0]
    N = len(X_train_combined)

    gmm_p = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
    gmm_b = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
    gmm_p.fit(X_pepper)
    gmm_b.fit(X_bg)

    def predict_gmm(x, margin=1.0):
        lp = gmm_p.score_samples(x) + np.log(len(X_pepper)/N)
        lb = gmm_b.score_samples(x) + np.log(len(X_bg)/N)
        return np.where(lp > lb + margin, 1, 0)




if __name__ == "__main__":
    main()
