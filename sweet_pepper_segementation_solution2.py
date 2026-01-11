from libs.logreg_utils import plot_precision_recall_curve, plot_segmentation_mask
from libs.data_utils import data_loader_q1, normalize_rgb_to_lab, process_input_image
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to segment')
parser.add_argument('--show_mask', action='store_true', help='Show segmentation mask visualization')
parser.add_argument('--show_metrics', action='store_true', help='Show precision-recall curves and validation metrics')
args = parser.parse_args()

def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()

    datasets = [("train", X_train), ("eval", X_eval), ("valid", X_valid)]
    lab_features = {}
    
    for name, data in tqdm(datasets, desc="LAB conversion"):
        lab_features[f"{name}_lab"] = normalize_rgb_to_lab(data)
    
    X_train_lab = lab_features["train_lab"]
    X_eval_lab = lab_features["eval_lab"]
    X_valid_lab = lab_features["valid_lab"]

    print("Training KNN classifier")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train_lab, y_train)

    if args.show_metrics:
        y_eval_pred = classifier.predict(X_eval_lab)
        y_valid_pred = classifier.predict(X_valid_lab)
        y_eval_scores = classifier.predict_proba(X_eval_lab)[:, 1]
        y_valid_scores = classifier.predict_proba(X_valid_lab)[:, 1]
        
        print("\n=== VALIDATION SET METRICS ===")
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_f1 = f1_score(y_valid, y_valid_pred)
        valid_precision = precision_score(y_valid, y_valid_pred)
        valid_recall = recall_score(y_valid, y_valid_pred)
        print(f"Accuracy: {valid_accuracy:.4f}")
        print(f"F1-Score: {valid_f1:.4f}")
        print(f"Precision: {valid_precision:.4f}")
        print(f"Recall: {valid_recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_valid, y_valid_pred, target_names=["Background", "Sweet Pepper"]))
        
        plot_precision_recall_curve(y_eval, y_eval_scores, title="Sweet Pepper Segmentation KNN - Precision-Recall Curve (Eval Set)")
        plot_precision_recall_curve(y_valid, y_valid_scores, title="Sweet Pepper Segmentation KNN - Precision-Recall Curve (Val Set)")

    y_pred_image, orig_shape, image = process_input_image(args.input_image, classifier, normalize_rgb_to_lab)
    percent_pepper = 100.0 * np.sum(y_pred_image == 1) / y_pred_image.size

    if args.show_mask:
        mask = y_pred_image.reshape(orig_shape)
        plot_segmentation_mask(mask, original_image=image)

    print(f"the model identified: {percent_pepper:.2f}% of total pixels as sweetpepper")

if __name__ == "__main__":
    main()