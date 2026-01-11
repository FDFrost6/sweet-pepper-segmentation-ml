from libs.logreg_utils import plot_precision_recall_curve, timing, plot_segmentation_mask
from libs.data_utils import data_loader_q1, normalize_rgb_to_lab, process_input_image
import argparse
import numpy as np
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to segment')
args = parser.parse_args()

def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()
    X_train_lab = normalize_rgb_to_lab(X_train)
    X_eval_lab = normalize_rgb_to_lab(X_eval)
    X_valid_lab = normalize_rgb_to_lab(X_valid)
    model = SVC(kernel='rbf', probability=True, gamma=0.01, C=1)
    model.fit(X_train_lab, y_train)
    y_eval_scores = model.predict_proba(X_eval_lab)[:, 1]
    y_valid_scores = model.predict_proba(X_valid_lab)[:, 1]
    plot_precision_recall_curve(y_eval, y_eval_scores, title="precision-recall curve (eval set)")
    plot_precision_recall_curve(y_valid, y_valid_scores, title="precision-recall curve (val set)")
    y_pred_img, orig_shape, image = process_input_image(args.input_image, model, normalize_rgb_to_lab)
    percent_pepper = 100.0 * np.sum(y_pred_img == 1) / y_pred_img.size
    #you can uncomment here to get a segmentation mask
    #mask = y_pred_img.reshape(orig_shape)
    #plot_segmentation_mask(mask, original_image=image)
    print(f"the model identified:{percent_pepper:.2f}% of total pixels as pepper:")

if __name__ == "__main__":
    main()

