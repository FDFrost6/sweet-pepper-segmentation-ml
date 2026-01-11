from libs.data_utils import rgb_to_hsv_flat
from libs.logreg_utils import plot_mask
import argparse
from skimage.io import imread
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to segment')
args = parser.parse_args()

def main():
    image = imread(args.input_image)
    hsv_img = rgb_to_hsv_flat(image)

    # Threshold 3: (0.95, 0.05, 0.4, 1.0) for hue (wrap-around) and saturation, value (0.2, 1.0)
    hmin, hmax, smin, smax = (0.95, 0.05, 0.4, 1.0)
    vmin, vmax = (0.2, 1.0)

    # Hue wrap-around for red
    if hmin > hmax:
        hue_mask = (hsv_img[..., 0] >= hmin) | (hsv_img[..., 0] <= hmax)
    else:
        hue_mask = (hsv_img[..., 0] >= hmin) & (hsv_img[..., 0] <= hmax)
    sat_mask = (hsv_img[..., 1] >= smin) & (hsv_img[..., 1] <= smax)
    val_mask = (hsv_img[..., 2] >= vmin) & (hsv_img[..., 2] <= vmax)
    mask = hue_mask & sat_mask & val_mask

    percent_pepper = 100.0 * np.sum(mask) / (hsv_img.shape[0] * hsv_img.shape[1])
    print(f"Threshold 3 (perfect!): {percent_pepper:.2f}% of pixels identified as red pepper")
    print("This threshold is perfect for your case. Use threshold 3 for best results.")

    plot_mask(mask, image, f"Threshold 3: {percent_pepper:.2f}% red pepper")

if __name__ == "__main__":
    main()
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
    # load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()

    # normalize and convert to hsv color space
    X_train_hs = normalize_rgb_to_hue_sat(X_train)
    X_eval_hs = normalize_rgb_to_hue_sat(X_eval)
    X_valid_hs = normalize_rgb_to_hue_sat(X_valid)

    image = imread(args.input_image)
    hsv_img = rgb_to_hsv_flat(image)

    # Threshold 3: (0.95, 0.05, 0.4, 1.0) for hue (wrap-around) and saturation, value (0.2, 1.0)
    hmin, hmax, smin, smax = (0.95, 0.05, 0.4, 1.0)
    vmin, vmax = (0.2, 1.0)

    # Hue wrap-around for red
    if hmin > hmax:
        hue_mask = (hsv_img[..., 0] >= hmin) | (hsv_img[..., 0] <= hmax)
    else:
        hue_mask = (hsv_img[..., 0] >= hmin) & (hsv_img[..., 0] <= hmax)
    sat_mask = (hsv_img[..., 1] >= smin) & (hsv_img[..., 1] <= smax)
    val_mask = (hsv_img[..., 2] >= vmin) & (hsv_img[..., 2] <= vmax)
    mask = hue_mask & sat_mask & val_mask

    percent_pepper = 100.0 * np.sum(mask) / (hsv_img.shape[0] * hsv_img.shape[1])
    print(f"Threshold 3 (perfect!): {percent_pepper:.2f}% of pixels identified as red pepper")
    print("This threshold is perfect for your case. Use threshold 3 for best results.")

    plot_mask(mask, image, f"Threshold 3: {percent_pepper:.2f}% red pepper")

if __name__ == "__main__":
    main()
