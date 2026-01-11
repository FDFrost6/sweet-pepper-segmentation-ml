from libs.pca_rf_utils import MyPCA
from libs.logreg_utils import plot_precision_recall_curve, timing
import argparse
import numpy as np
import pickle
import time
from skimage import color
from skimage.feature import local_binary_pattern
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report, accuracy_score
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='Path to input image to classify')
args = parser.parse_args()

def load_q2_data():
    # Load sweet pepper (1) and background (0) data for Q2
    with open('data/Q2_BG_dict.pkl', 'rb') as f:
        bg = pickle.load(f)
    with open('data/Q2_SP_dict.pkl', 'rb') as f:
        sp = pickle.load(f)
    # Stack and label
    X_train = np.concatenate([np.stack(bg['train']), np.stack(sp['train'])], axis=0)
    y_train = np.concatenate([np.zeros(len(bg['train'])), np.ones(len(sp['train']))])
    X_eval = np.concatenate([np.stack(bg['evaluation']), np.stack(sp['evaluation'])], axis=0)
    y_eval = np.concatenate([np.zeros(len(bg['evaluation'])), np.ones(len(sp['evaluation']))])
    X_valid = np.concatenate([np.stack(bg['validation']), np.stack(sp['validation'])], axis=0)
    y_valid = np.concatenate([np.zeros(len(bg['validation'])), np.ones(len(sp['validation']))])
    return X_train, y_train, X_eval, y_eval, X_valid, y_valid

def compute_lab_histograms(X, bins=16, range_L=(0, 100), range_ab=(-128, 127)):
    """
    Compute concatenated histograms for each Lab channel for all images in X.
    X: (num_samples, H, W, 3) in Lab color space.
    Returns: (num_samples, bins*3) array.
    """
    X = np.asarray(X)
    n = X.shape[0]
    feats = []
    for i in range(n):
        img = X[i]
        L_hist, _ = np.histogram(img[..., 0], bins=bins, range=range_L, density=True)
        a_hist, _ = np.histogram(img[..., 1], bins=bins, range=range_ab, density=True)
        b_hist, _ = np.histogram(img[..., 2], bins=bins, range=range_ab, density=True)
        feats.append(np.concatenate([L_hist, a_hist, b_hist]))
    return np.array(feats)

def compute_spatial_histograms(X, bins=8, grid_size=2, range_L=(0, 100), range_ab=(-128, 127)):
    """
    Compute spatial color histograms for each Lab channel in grid patches.
    Returns: (num_samples, bins*3*grid_size*grid_size) array.
    """
    X = np.asarray(X)
    n, h, w, c = X.shape
    feats = []
    h_step = h // grid_size
    w_step = w // grid_size
    for i in range(n):
        img = X[i]
        patch_feats = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                patch = img[gy*h_step:(gy+1)*h_step, gx*w_step:(gx+1)*w_step, :]
                L_hist, _ = np.histogram(patch[..., 0], bins=bins, range=range_L, density=True)
                a_hist, _ = np.histogram(patch[..., 1], bins=bins, range=range_ab, density=True)
                b_hist, _ = np.histogram(patch[..., 2], bins=bins, range=range_ab, density=True)
                patch_feats.extend([L_hist, a_hist, b_hist])
        feats.append(np.concatenate(patch_feats))
    return np.array(feats)

def compute_lbp_features(X, P=8, R=1, bins=16):
    """
    Compute LBP histograms for the L channel of Lab images.
    Returns: (num_samples, bins) array.
    """
    X = np.asarray(X)
    n = X.shape[0]
    feats = []
    for i in range(n):
        img = X[i]
        L = img[..., 0]
        # Convert L channel to uint8 for stable LBP
        L_uint8 = np.clip((L / 100.0) * 255, 0, 255).astype(np.uint8)
        lbp = local_binary_pattern(L_uint8, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, P+2), density=True)
        feats.append(hist)
    return np.array(feats)

def extract_hsv_histogram_skimage(img, h_bins=16, s_bins=8, v_bins=8):
    """
    Compute a 3D HSV histogram for a single image using skimage.
    img: (H, W, 3) in RGB [0,1] or [0,255]
    Returns: (h_bins * s_bins * v_bins,) normalized histogram
    """
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    hsv = color.rgb2hsv(arr)
    h = hsv[..., 0] * 180  # skimage hue is [0,1], OpenCV is [0,180]
    s = hsv[..., 1] * 255
    v = hsv[..., 2] * 255
    hist, _ = np.histogramdd(
        sample=[h.ravel(), s.ravel(), v.ravel()],
        bins=[h_bins, s_bins, v_bins],
        range=[[0,180], [0,256], [0,256]]
    )
    hist = hist.flatten()
    hist /= (hist.sum() + 1e-6)
    return hist

def compute_hsv_histograms(X, h_bins=16, s_bins=8, v_bins=8):
    """
    Compute HSV histograms for a batch of images.
    X: (num_samples, H, W, 3) in RGB [0,1] or [0,255]
    Returns: (num_samples, h_bins*s_bins*v_bins)
    """
    feats = []
    for i in range(X.shape[0]):
        feats.append(extract_hsv_histogram_skimage(X[i], h_bins, s_bins, v_bins))
    return np.array(feats)

@timing
def main():
    # Load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()

    # Normalize data to [0, 1]
    X_train = X_train.astype(np.float32)
    X_eval = X_eval.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    X_train /= 255.0
    X_eval /= 255.0
    X_valid /= 255.0

    # Convert to Lab color space
    X_train_lab = color.rgb2lab(X_train)
    X_eval_lab = color.rgb2lab(X_eval)
    X_valid_lab = color.rgb2lab(X_valid)

    # Compute spatial color histograms (e.g., 2x2 grid, 8 bins per channel)
    X_train_spatial_hist = compute_spatial_histograms(X_train_lab, bins=8, grid_size=2)
    X_eval_spatial_hist = compute_spatial_histograms(X_eval_lab, bins=8, grid_size=2)
    X_valid_spatial_hist = compute_spatial_histograms(X_valid_lab, bins=8, grid_size=2)

    # Compute HSV histogram features
    X_train_hsv_hist = compute_hsv_histograms(X_train, h_bins=8, s_bins=4, v_bins=4)
    X_eval_hsv_hist = compute_hsv_histograms(X_eval, h_bins=8, s_bins=4, v_bins=4)
    X_valid_hsv_hist = compute_hsv_histograms(X_valid, h_bins=8, s_bins=4, v_bins=4)

    # Compute LBP features (from L channel of Lab)
    X_train_lab = color.rgb2lab(X_train)
    X_eval_lab = color.rgb2lab(X_eval)
    X_valid_lab = color.rgb2lab(X_valid)
    X_train_lbp = compute_lbp_features(X_train_lab, P=8, R=1, bins=16)
    X_eval_lbp = compute_lbp_features(X_eval_lab, P=8, R=1, bins=16)
    X_valid_lbp = compute_lbp_features(X_valid_lab, P=8, R=1, bins=16)

    # Concatenate HSV histogram and LBP features
    X_train_combined = np.concatenate([X_train_hsv_hist, X_train_lbp], axis=1)
    X_eval_combined = np.concatenate([X_eval_hsv_hist, X_eval_lbp], axis=1)
    X_valid_combined = np.concatenate([X_valid_hsv_hist, X_valid_lbp], axis=1)

    # Apply PCA before SVM
    n_components = 20
    pca = MyPCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_combined)
    X_eval_pca = pca.transform(X_eval_combined)
    X_valid_pca = pca.transform(X_valid_combined)

    # Train SVM classifier on PCA features
    model = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, class_weight="balanced")
    model.fit(X_train_pca, y_train)

    # Precision-recall curves for combined features
    y_eval_scores = model.predict_proba(X_eval_pca)[:, 1]
    y_valid_scores = model.predict_proba(X_valid_pca)[:, 1]
    plot_precision_recall_curve(y_eval, y_eval_scores, title="precision-recall curve (eval set, SVM PCA HSV Hist + LBP)")
    plot_precision_recall_curve(y_valid, y_valid_scores, title="precision-recall curve (val set, SVM PCA HSV Hist + LBP)")

    # F1, precision, recall, confusion matrix, and classification report for validation set (combined)
    y_valid_pred = model.predict(X_valid_pca)
    f1 = f1_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    acc = accuracy_score(y_valid, y_valid_pred)
    cm = confusion_matrix(y_valid, y_valid_pred)
    print("\nConfusion Matrix (Validation, HSV+LBP PCA):")
    print("           Not Pepper  Sweet Pepper")
    print(f"Not Pepper   {cm[0,0]:10d} {cm[0,1]:12d}")
    print(f"Sweet Pepper {cm[1,0]:10d} {cm[1,1]:12d}")
    print("\n=== Validation (SVM PCA HSV Hist + LBP) ===")
    print(classification_report(y_valid, y_valid_pred, target_names=["Not Pepper", "Sweet Pepper"]))
    print(f"Accuracy: {acc:.6f}")

    # Process input image
    import skimage.io
    from skimage.transform import resize
    image = skimage.io.imread(args.input_image)
    if image.max() > 1.0:
        image = image / 255.0
    image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
    image_hsv_hist = compute_hsv_histograms(image_resized[None, ...], h_bins=8, s_bins=4, v_bins=4)
    image_lab = color.rgb2lab(image_resized[None, ...])
    image_lbp = compute_lbp_features(image_lab, P=8, R=1, bins=16)
    image_combined = np.concatenate([image_hsv_hist, image_lbp], axis=1)
    image_pca = pca.transform(image_combined)
    pred_proba = model.predict_proba(image_pca)[0, 1]
    print(f"Predicted probability for sweet pepper: {pred_proba:.4f}")
    threshold = 0.5
    pred = int(pred_proba > threshold)
    print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")

if __name__ == "__main__":
    main()

