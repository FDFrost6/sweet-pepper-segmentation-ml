from libs.svm_utils import extract_features_hsv_lbp, process_input_image
from libs.data_utils import load_q2_data
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='path to input image')
args = parser.parse_args()

def main():
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()
    # Feature extraction
    X_train_feat = extract_features_hsv_lbp(X_train)
    X_eval_feat = extract_features_hsv_lbp(X_eval)
    X_valid_feat = extract_features_hsv_lbp(X_valid)

    # SVMs (no PCA)
    print("=== SVM WITHOUT PCA ===")
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf.fit(X_train_feat, y_train)
    y_valid_pred = clf.predict(X_valid_feat)
    acc = accuracy_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    cm = confusion_matrix(y_valid, y_valid_pred)
    print("\nConfusion Matrix (Validation):")
    print(cm)
    print(classification_report(y_valid, y_valid_pred, target_names=["Not Pepper", "Sweet Pepper"]))
    print(f"Accuracy: {acc:.6f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    process_input_image(args.input_image, None, clf)  # None for pca

    # SVMs (with PCA)
    print("\n=== SVM WITH PCA ===")
    pca = PCA(n_components=20, random_state=42)
    X_train_pca = pca.fit_transform(X_train_feat)
    X_eval_pca = pca.transform(X_eval_feat)
    X_valid_pca = pca.transform(X_valid_feat)
    clf_pca = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf_pca.fit(X_train_pca, y_train)
    y_valid_pred_pca = clf_pca.predict(X_valid_pca)
    acc_pca = accuracy_score(y_valid, y_valid_pred_pca)
    precision_pca = precision_score(y_valid, y_valid_pred_pca)
    recall_pca = recall_score(y_valid, y_valid_pred_pca)
    f1_pca = f1_score(y_valid, y_valid_pred_pca)
    cm_pca = confusion_matrix(y_valid, y_valid_pred_pca)
    print("\nConfusion Matrix (Validation, PCA):")
    print(cm_pca)
    print(classification_report(y_valid, y_valid_pred_pca, target_names=["Not Pepper", "Sweet Pepper"]))
    print(f"Accuracy: {acc_pca:.6f}  Precision: {precision_pca:.4f}  Recall: {recall_pca:.4f}  F1: {f1_pca:.4f}")
    process_input_image(args.input_image, pca, clf_pca)

if __name__ == "__main__":
    main()
