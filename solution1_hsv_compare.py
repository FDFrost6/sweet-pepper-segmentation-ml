from libs.logreg_utils import MyLogReg, plot_precision_recall_curve, plot_mask
from libs.data_utils import data_loader_q1, rgb_to_hsv_flat
import argparse
import numpy as np
from skimage.io import imread

def normalize_rgb_to_hue_sat(input_array):
    """
    Converts RGB to HSV and returns hue and saturation channels as features.
    """
    arr = np.array(input_array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    original_shape = arr.shape
    arr_reshaped = arr.reshape(-1, 1, 3)
    hsv = rgb_to_hsv_flat(arr_reshaped)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    hs = np.concatenate([hue, sat], axis=-1)
    return hs.reshape(original_shape[:-1] + (2,))

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to segment')
args = parser.parse_args()

def main():
    # load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()

    # convert all data to HSV hue+saturation features
    X_train_hs = normalize_rgb_to_hue_sat(X_train)
    X_eval_hs = normalize_rgb_to_hue_sat(X_eval)
    X_valid_hs = normalize_rgb_to_hue_sat(X_valid)

    # train logistic regression on HSV features
    model = MyLogReg(learning_rate=0.0001, num_iterations=50)
    model.fit(X_train_hs, y_train)

    # plot precision-recall curves for eval and val
    y_eval_scores = model.predict_proba(X_eval_hs)[:, 1]
    y_valid_scores = model.predict_proba(X_valid_hs)[:, 1]
    plot_precision_recall_curve(y_eval, y_eval_scores, title="precision-recall curve (eval set, HSV)")
    plot_precision_recall_curve(y_valid, y_valid_scores, title="precision-recall curve (val set, HSV)")

    # process input image for model prediction
    image = imread(args.input_image)
    hsv_img = rgb_to_hsv_flat(image)
    image_hs = normalize_rgb_to_hue_sat(image)
    y_pred_img = model.predict(image_hs)
    orig_shape = image.shape[:2]
    mask_model = y_pred_img.reshape(orig_shape)
    percent_pepper_model = 100.0 * np.sum(mask_model == 1) / mask_model.size
    plot_mask(mask_model == 1, image, f"MyLogReg HSV: {percent_pepper_model:.2f}% red pepper")
    print(f"MyLogReg HSV: {percent_pepper_model:.2f}% of pixels identified as red pepper")

    # manual threshold (Threshold 3)
    hmin, hmax, smin, smax = (0.95, 0.05, 0.4, 1.0)
    vmin, vmax = (0.2, 1.0)
    if hmin > hmax:
        hue_mask = (hsv_img[..., 0] >= hmin) | (hsv_img[..., 0] <= hmax)
    else:
        hue_mask = (hsv_img[..., 0] >= hmin) & (hsv_img[..., 0] <= hmax)
    sat_mask = (hsv_img[..., 1] >= smin) & (hsv_img[..., 1] <= smax)
    val_mask = (hsv_img[..., 2] >= vmin) & (hsv_img[..., 2] <= vmax)
    mask_thresh = hue_mask & sat_mask & val_mask
    percent_pepper_thresh = 100.0 * np.sum(mask_thresh) / mask_thresh.size
    plot_mask(mask_thresh, image, f"Manual Threshold 3: {percent_pepper_thresh:.2f}% red pepper")
    print(f"Manual Threshold 3: {percent_pepper_thresh:.2f}% of pixels identified as red pepper")

if __name__ == "__main__":
    main()
