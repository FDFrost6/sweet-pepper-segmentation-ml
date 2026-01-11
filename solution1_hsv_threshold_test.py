import argparse
import numpy as np
from skimage import color
from skimage.io import imread
import matplotlib.pyplot as plt
from libs.data_utils import process_input_image

def rgb_to_hsv_flat(image):
    arr = np.array(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    orig_shape = arr.shape
    arr_reshaped = arr.reshape(-1, 1, 3)
    hsv = color.rgb2hsv(arr_reshaped)
    return hsv.reshape(orig_shape)

def apply_thresholds(hsv_img, thresholds, v_thresholds=None):
    masks = []
    for i, (hmin, hmax, smin, smax) in enumerate(thresholds):
        # Hue wrap-around for red (e.g., hmin > hmax)
        if hmin > hmax:
            hue_mask = (hsv_img[..., 0] >= hmin) | (hsv_img[..., 0] <= hmax)
        else:
            hue_mask = (hsv_img[..., 0] >= hmin) & (hsv_img[..., 0] <= hmax)
        sat_mask = (hsv_img[..., 1] >= smin) & (hsv_img[..., 1] <= smax)
        mask = hue_mask & sat_mask
        # Optionally add value threshold
        if v_thresholds is not None:
            vmin, vmax = v_thresholds[i]
            val_mask = (hsv_img[..., 2] >= vmin) & (hsv_img[..., 2] <= vmax)
            mask = mask & val_mask
        masks.append(mask)
    return masks

def plot_mask(mask, image, title):
    overlay = image.astype(np.float32)
    if overlay.max() > 1.0:
        overlay = overlay / 255.0
    overlay = overlay.copy()
    overlay[mask] = [0, 0, 0]
    plt.figure(figsize=(6, 8))
    plt.imshow(overlay)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to segment')
args = parser.parse_args()

def main():
    image = imread(args.input_image)
    hsv_img = rgb_to_hsv_flat(image)

    # List of (hue_min, hue_max, sat_min, sat_max) thresholds to try
    thresholds = [
        (0.94, 0.06, 0.5, 1.0),   # slightly wider hue
        (0.96, 0.04, 0.6, 1.0),   # narrower hue, higher sat
        (0.95, 0.05, 0.4, 1.0),   # lower sat
        (0.0, 0.1, 0.4, 1.0),     # wider hue, lower sat
        (0.9, 0.1, 0.3, 1.0),     # even wider, lower sat
    ]
    v_thresholds = [
        (0.2, 1.0),
        (0.2, 1.0),
        (0.2, 1.0),
        (0.2, 1.0),
        (0.2, 1.0),
    ]

    masks = apply_thresholds(hsv_img, thresholds, v_thresholds)
    total_pixels = hsv_img.shape[0] * hsv_img.shape[1]

    for i, mask in enumerate(masks):
        percent_pepper = 100.0 * np.sum(mask) / total_pixels
        if i == 0:
            print(f"Threshold {i+1} (recommended): {percent_pepper:.2f}% of pixels identified as red pepper")
            print("This threshold gives minimal yellow misclassification but may miss some red spots.")
            print("You can fine-tune the hue/sat range in threshold 1 for even better results.")
        else:
            print(f"Threshold {i+1}: {percent_pepper:.2f}% of pixels identified as red pepper")
        plot_mask(mask, image, f"Threshold {i+1}: {percent_pepper:.2f}% red pepper")

if __name__ == "__main__":
    main()
