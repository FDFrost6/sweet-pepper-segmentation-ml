import argparse
import numpy as np
from skimage import io, color, feature
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pickle

def extract_lab_histogram(img, l_bins=8, a_bins=4, b_bins=4):
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    lab = color.rgb2lab(arr)
    l = lab[..., 0]  # [0, 100]
    a = lab[..., 1]  # [-128, 127]
    b = lab[..., 2]  # [-128, 127]
    hist, _ = np.histogramdd(
        sample=[l.ravel(), a.ravel(), b.ravel()],
        bins=[l_bins, a_bins, b_bins],
        range=[[0, 100], [-128, 128], [-128, 128]]
    )
    hist = hist.flatten()
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_lbp_histogram(img, P=8, R=1):
    gray = rgb2gray(img)
    gray_uint8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(gray_uint8, P, R, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, 59+1), range=(0,59))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(X):
    feats = []
    for img in X:
        lab_hist = extract_lab_histogram(img, l_bins=8, a_bins=4, b_bins=4)
        lbp_hist = extract_lbp_histogram(img, P=8, R=1)
        feats.append(np.concatenate([lab_hist, lbp_hist]))
    return np.array(feats)

def load_q2_data():
    with open('data/Q2_BG_dict.pkl', 'rb') as f:
        bg = pickle.load(f)
    with open('data/Q2_SP_dict.pkl', 'rb') as f:
        sp = pickle.load(f)
    X_train = np.concatenate([np.stack(bg['train']), np.stack(sp['train'])], axis=0)
    y_train = np.concatenate([np.zeros(len(bg['train'])), np.ones(len(sp['train']))])
    X_eval = np.concatenate([np.stack(bg['evaluation']), np.stack(sp['evaluation'])], axis=0)
    y_eval = np.concatenate([np.zeros(len(bg['evaluation'])), np.ones(len(sp['evaluation']))])
    X_valid = np.concatenate([np.stack(bg['validation']), np.stack(sp['validation'])], axis=0)
    y_valid = np.concatenate([np.zeros(len(bg['validation'])), np.ones(len(sp['validation']))])
    return X_train, y_train, X_eval, y_eval, X_valid, y_valid

def plot_precision_recall(y_true, y_scores, title="precision-recall curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.step(recall, precision, where='post', label=f'AP={avg_precision:.3f}')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str, help='Path to input image to classify')
    args = parser.parse_args()

    # Load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()

    # Feature extraction
    print("Extracting features...")
    X_train_feat = extract_features(X_train)
    X_eval_feat = extract_features(X_eval)
    X_valid_feat = extract_features(X_valid)

    # PCA
    print("Applying PCA...")
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_feat)
    X_eval_pca = pca.transform(X_eval_feat)
    X_valid_pca = pca.transform(X_valid_feat)

    # Classifier
    print("Training SVM...")
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    clf.fit(X_train_pca, y_train)

    # Validation set threshold tuning
    y_valid_scores = clf.predict_proba(X_valid_pca)[:, 1]
    plot_precision_recall(y_valid, y_valid_scores, title="precision-recall curve (val set, SVM LAB+LBP PCA)")

    # Find threshold for best F1
    precision, recall, thresholds = precision_recall_curve(y_valid, y_valid_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Suggested threshold for best F1: {best_threshold:.3f}")

    # Validation set metrics at best threshold
    y_valid_pred = (y_valid_scores > best_threshold).astype(int)
    print("\nConfusion Matrix (Validation):")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, target_names=["Not Pepper", "Sweet Pepper"]))

    # Process input image
    image = io.imread(args.input_image)
    if image.max() > 1.0:
        image = image / 255.0
    from skimage.transform import resize
    image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
    image_feat = extract_features([image_resized])
    image_pca = pca.transform(image_feat)
    pred_proba = clf.predict_proba(image_pca)[0, 1]
    print(f"Predicted probability for sweet pepper: {pred_proba:.4f}")
    pred = int(pred_proba > best_threshold)
    print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")

def main_no_pca():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str, help='Path to input image to classify')
    args = parser.parse_args()

    # Load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()

    # Feature extraction
    print("Extracting features (no PCA)...")
    X_train_feat = extract_features(X_train)
    X_eval_feat = extract_features(X_eval)
    X_valid_feat = extract_features(X_valid)

    # Classifier
    print("Training SVM (no PCA)...")
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    clf.fit(X_train_feat, y_train)

    # Validation set threshold tuning
    y_valid_scores = clf.predict_proba(X_valid_feat)[:, 1]
    plot_precision_recall(y_valid, y_valid_scores, title="precision-recall curve (val set, SVM LAB+LBP NO PCA)")

    # Find threshold for best F1
    precision, recall, thresholds = precision_recall_curve(y_valid, y_valid_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Suggested threshold for best F1: {best_threshold:.3f}")

    # Validation set metrics at best threshold
    y_valid_pred = (y_valid_scores > best_threshold).astype(int)
    print("\nConfusion Matrix (Validation, no PCA):")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, target_names=["Not Pepper", "Sweet Pepper"]))

    # Process input image
    image = io.imread(args.input_image)
    if image.max() > 1.0:
        image = image / 255.0
    from skimage.transform import resize
    image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
    image_feat = extract_features([image_resized])
    pred_proba = clf.predict_proba(image_feat)[0, 1]
    print(f"Predicted probability for sweet pepper: {pred_proba:.4f}")
    pred = int(pred_proba > best_threshold)
    print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")

if __name__ == "__main__":
    main()
    # To run without PCA, comment out the above and uncomment below:
    # main_no_pca()
