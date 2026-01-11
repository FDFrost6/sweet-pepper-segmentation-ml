# =========================
# Simplified script for image segmentation with timing and pepper percentage
# =========================
import argparse
import numpy as np
import time
from libs.data_utils import simple_data_loader_Q1
from libs.knn_utils_test import train_custom_knn, convert_color_space, manual_normalize, print_knn_metrics
from skimage.io import imread, imsave

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str, help='Path to input image to segment')
    args = parser.parse_args()


    # Load and label data (only bg and red)
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = simple_data_loader_Q1()
    # Use only 1/4 of the training data
    n_subset = X_train.shape[0] // 4
    X_train = X_train[:n_subset]
    y_train = y_train[:n_subset]

    # Preprocess (normalize, color space LAB only)
    X_train = manual_normalize(convert_color_space(X_train, 'lab')).astype(np.float32)


    # Train KNN (self-coded, no speedups)
    knn = train_custom_knn(X_train, y_train, n_neighbors=5)

    # Preprocess eval and val sets (lab color space)
    X_eval = manual_normalize(convert_color_space(X_eval, 'lab')).astype(np.float32)
    X_valid = manual_normalize(convert_color_space(X_valid, 'lab')).astype(np.float32)

    # Predict on eval and val
    y_eval_pred = knn.predict(X_eval)
    y_valid_pred = knn.predict(X_valid)


    # Print accuracy, confusion matrix, precision, recall
    print_knn_metrics(y_eval, y_eval_pred, name="Evaluation Set")
    print_knn_metrics(y_valid, y_valid_pred, name="Validation Set")

    # Segment input image and time the process
    image = imread(args.input_image)
    if image.max() > 1.0:
        image = image / 255.0
    orig_shape = image.shape[:2]
    pixels = image.reshape(-1, 3)
    pixels_proc = manual_normalize(convert_color_space(pixels, 'lab')).astype(np.float32)


    # Use tqdm for live progress during segmentation
    from tqdm import tqdm
    start = time.time()
    # If knn.predict supports batch, wrap with tqdm
    batch_size = 1000
    n_pixels = pixels_proc.shape[0]
    y_pred = np.zeros(n_pixels, dtype=np.int32)
    for i in tqdm(range(0, n_pixels, batch_size), desc='Segmenting image'):
        batch = pixels_proc[i:i+batch_size]
        y_pred[i:i+batch_size] = knn.predict(batch)
    end = time.time()

    segmentation = y_pred.reshape(orig_shape)

    # Calculate percentage of pepper pixels (not bg)
    n_total = segmentation.size
    n_pepper = np.count_nonzero(segmentation != 0)
    percent_pepper = 100.0 * n_pepper / n_total

    # Save mask as image (color mapping for bg and red only)
    mask_rgb = np.zeros((orig_shape[0], orig_shape[1], 3), dtype=np.uint8)
    class_colors = {0: [0,0,255], 1: [255,0,0]}  # bg: blue, red: red
    for label, color in class_colors.items():
        mask_rgb[segmentation == label] = color
    imsave('segmentation_mask.png', mask_rgb)

    print(f"Time taken to process image: {end-start:.2f} seconds")
    print(f"Percentage of pepper pixels (not bg): {percent_pepper:.2f}%")
    print("Segmentation mask saved as segmentation_mask.png")

if __name__ == '__main__':
    main()
