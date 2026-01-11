from libs.logreg_utils import MyLogReg, timing
from libs.data_utils import data_loader_q1_extra, normalize_rgb_to_lab
import argparse
import numpy as np
from skimage.io import imread
import time

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to classify')
args = parser.parse_args()

@timing
def main():
    # load data with yellow as a third class
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1_extra()

    # convert all data to lab color space
    X_train_lab = normalize_rgb_to_lab(X_train)
    X_eval_lab = normalize_rgb_to_lab(X_eval)
    X_valid_lab = normalize_rgb_to_lab(X_valid)

    # train multinomial logistic regression (one-vs-rest)
    model = MyLogReg(learning_rate=1e-2, num_iterations=200)
    model.fit(X_train_lab, y_train)

    # process input image
    image = imread(args.input_image)
    image_rgb = image[..., :3]  # ensure only RGB channels
    orig_shape = image_rgb.shape[:2]
    image_lab = normalize_rgb_to_lab(image_rgb.reshape(-1, 3))
    y_pred_img = model.predict(image_lab)
    mask = y_pred_img.reshape(orig_shape)

    # calculate percentages
    total_pixels = y_pred_img.size
    percent_red = 100.0 * np.sum(y_pred_img == 1) / total_pixels
    percent_yellow = 100.0 * np.sum(y_pred_img == 2) / total_pixels
    percent_pepper = percent_red + percent_yellow

    # threshold logic
    if percent_pepper > 10.0:
        if percent_red >= percent_yellow:
            result = "red/pepper"
        else:
            result = "yellow/pepper"
        print(f"the image is classified as: {result} (more than 10% pepper pixels)")
    else:
        print("the image is classified as: background (less than or equal to 20% pepper pixels)")

    print(f"percentage of red pixels: {percent_red:.2f}%")
    print(f"percentage of yellow pixels: {percent_yellow:.2f}%")

if __name__ == "__main__":
    main()