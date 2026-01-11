from libs.logreg_utils import plot_segmentation_mask
from libs.data_utils import data_loader_q1_extra, normalize_rgb_to_lab, process_input_image
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to segment')
parser.add_argument('--show_mask', action='store_true', help='Show segmentation mask visualization')
parser.add_argument('--show_metrics', action='store_true', help='Show validation metrics and performance')
args = parser.parse_args()

def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1_extra()
    
    X_train_lab = normalize_rgb_to_lab(X_train)

    print("Training KNN classifier")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train_lab, y_train)

    if args.show_metrics:
        X_valid_lab = normalize_rgb_to_lab(X_valid)
        y_valid_pred = classifier.predict(X_valid_lab)
        
        print("\n=== VALIDATION SET METRICS ===")
        accuracy = accuracy_score(y_valid, y_valid_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_valid, y_valid_pred, target_names=["Background", "Red Sweetpepper", "Yellow Sweetpepper"]))

    y_pred_image, orig_shape, image = process_input_image(args.input_image, classifier, normalize_rgb_to_lab)
    total_pixels = y_pred_image.size

    if args.show_mask:
        mask = y_pred_image.reshape(orig_shape)
        plot_segmentation_mask(mask, original_image=image)

    percent_bg = 100.0 * np.sum(y_pred_image == 0) / total_pixels
    percent_red = 100.0 * np.sum(y_pred_image == 1) / total_pixels
    percent_yellow = 100.0 * np.sum(y_pred_image == 2) / total_pixels
    print(f"percentage of background pixels: {percent_bg:.2f}%")
    print(f"percentage of red/sweetpepper pixels: {percent_red:.2f}%")
    print(f"percentage of yellow/sweetpepper pixels: {percent_yellow:.2f}%")
    
    percentages = [percent_bg, percent_red, percent_yellow]
    class_names = ["background", "red sweetpepper", "yellow sweetpepper"]
    output_class = np.argmax(percentages)
    print(f"the image is probably: {class_names[output_class]}")

if __name__ == "__main__":
    main()