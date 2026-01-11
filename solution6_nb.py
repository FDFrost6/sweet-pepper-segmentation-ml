from libs.data_utils import load_q2_data, rgb_to_hsv_flat
import argparse
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

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
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(flatten_hsv_images(X_train))
    X_valid_flat = scaler.transform(flatten_hsv_images(X_valid))

    # Fit Naive Bayes classifier
    nb = GaussianNB()
    nb.fit(X_train_flat, y_train)

    # Validation metrics
    y_valid_pred = nb.predict(X_valid_flat)
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
    hsv = rgb_to_hsv_flat(image_resized)
    image_flat = hsv.flatten().reshape(1, -1)
    image_flat = scaler.transform(image_flat)
    pred = nb.predict(image_flat)[0]
    print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")

if __name__ == "__main__":
    main()
