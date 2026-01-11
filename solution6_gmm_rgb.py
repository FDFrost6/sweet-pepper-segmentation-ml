from libs.data_utils import load_q2_data
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='path to input image')
args = parser.parse_args()

def flatten_rgb_images(X):
    """
    Converts a batch of RGB images to flattened 2D array (num_samples, num_pixels * num_channels).
    """
    rgb_images = []
    for img in X:
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        rgb_images.append(arr.flatten())
    return np.array(rgb_images)

def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(flatten_rgb_images(X_train))
    X_valid_flat = scaler.transform(flatten_rgb_images(X_valid))

    # Split training data by class
    X_pepper = X_train_flat[y_train == 1]
    X_bg = X_train_flat[y_train == 0]
    N = len(X_train_flat)

    # Fit GMMs
    gmm_p = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
    gmm_b = GaussianMixture(n_components=3, covariance_type="diag", random_state=42)
    gmm_p.fit(X_pepper)
    gmm_b.fit(X_bg)

    def predict_gmm(x, margin=1.0):
        lp = gmm_p.score_samples(x) + np.log(len(X_pepper)/N)
        lb = gmm_b.score_samples(x) + np.log(len(X_bg)/N)
        return np.where(lp > lb + margin, 1, 0)

    # Validation metrics
    y_valid_pred = predict_gmm(X_valid_flat)
    print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_valid, y_valid_pred))
    print("Classification Report:\n", classification_report(y_valid, y_valid_pred, target_names=["Not Pepper", "Sweet Pepper"]))

    # Process input image
    from skimage.io import imread
    from skimage.transform import resize
    image = imread(args.input_image)
    if image.max() > 1.0:
        image = image / 255.0
    image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
    arr = np.array(image_resized, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    image_flat = arr.flatten().reshape(1, -1)
    image_flat = scaler.transform(image_flat)
    pred = predict_gmm(image_flat)[0]
    print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")

if __name__ == "__main__":
    main()
