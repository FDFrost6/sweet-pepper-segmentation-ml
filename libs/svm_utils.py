import numpy as np
from skimage import color, feature 
from skimage.color import rgb2gray
from skimage.transform import resize
from libs.logreg_utils import timing
from skimage.io import imread
from libs.data_utils import rgb_to_hsv_flat


def extract_hsv_histogram_skimage(img, h_bins=8, s_bins=4, v_bins=4):
    """
    extracts a normalized HSV histogram from an image using skimage.
    args:input image, shape (..., 3), values in [0,255] or [0,1].
        h_bins, s_bins, v_bins (int): number of bins for H, S, V channels.
    returns:flattened and normalized histogram vector.
    """
    array = np.array(img, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    hsv = color.rgb2hsv(array)
    h = hsv[..., 0] * 180
    s = hsv[..., 1] * 255
    v = hsv[..., 2] * 255
    histogram, _ = np.histogramdd(sample=[h.ravel(), s.ravel(), v.ravel()], bins=[h_bins, s_bins, v_bins], range=[[0,180], [0,256], [0,256]])
    histogram = histogram.flatten()
    histogram /= (histogram.sum() + 1e-6)
    return histogram

def compute_hsv_histograms(image_batch, h_bins=8, s_bins=4, v_bins=4):
    """
    computes HSV histograms for a batch of images.
    args:batch of images, shape (N, H, W, 3).
         h_bins, s_bins, v_bins: number of bins for H, S, V channels.
    returns:feature array of shape (N, h_bins*s_bins*v_bins) with histograms.
    """
    image_batch = np.asarray(image_batch)
    features = []
    for i in range(image_batch.shape[0]):
        features.append(extract_hsv_histogram_skimage(image_batch[i], h_bins, s_bins, v_bins))
    return np.array(features)

def compute_lbp_features_gray(image_batch, P=8, R=1, bins=59):
    """
    computes LBP histograms for a batch of grayscale images.
    args: batch of images, shape (N, H, W, 3).
        P: number of circularly symmetric neighbour set points.
        R: radius of circle.
        bins: number of histogram bins.
    returns: array of shape (N, bins) with lbp histograms.
    """
    image_batch = np.asarray(image_batch)
    features = []
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        gray = rgb2gray(image)
        gray_uint8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
        lbp = feature.local_binary_pattern(gray_uint8, P, R, method='uniform')
        histogram, _ = np.histogram(lbp, bins=np.arange(0, bins+1), range=(0, bins))
        histogram = histogram.astype(np.float32)
        histogram /= (histogram.sum() + 1e-6)
        features.append(histogram)
    return np.array(features)

def extract_features_hsv_lbp(image_batch):
    """
    extracts hsv and lbp features for a image batch
    args: batch of images shape (N, H, W, 3).
    returns: array of shape (N, feature_dim) with combined features.
    """
    image_batch = np.asarray(image_batch)
    hsv_histogram = compute_hsv_histograms(image_batch, h_bins=8, s_bins=4, v_bins=4)
    lbp_histogram = compute_lbp_features_gray(image_batch, P=8, R=1, bins=59)
    return np.concatenate([hsv_histogram, lbp_histogram], axis=1)

@timing
def process_input_imageQ2(image_path, classifier, threshold=0.5):
    """
    loads and resizes an image, extracts features, predicts probability using a classifier
    and prints the predicted class.
    args:image_path to input image.
         trained classifier
         decision threshold for class prediction.
    """
    image = imread(image_path)
    if image.max() > 1.0:
        image = image / 255.0
    image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
    image_feat = extract_features_hsv_lbp([image_resized])
    proba = classifier.predict_proba(image_feat)[0, 1]
    print(f"Predicted probability for sweet pepper: {proba:.4f}")
    pred = int(proba > threshold)
    print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")


# def process_input_imageQ2_p2(image_path, classifier, pca, threshold=0.5):
#     """
#     Loads and resizes an image, converts to HSV using rgb_to_hsv_flat, flattens, applies PCA, predicts probability using a classifier.
#     Prints the predicted class.
#     """
#     image = imread(image_path)
#     if image.max() > 1.0:
#         image = image / 255.0
#     image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
#     hsv = rgb_to_hsv_flat(image_resized)
#     image_flat = hsv.flatten().reshape(1, -1)
#     image_pca = pca.transform(image_flat)
#     proba = classifier.predict_proba(image_pca)[0, 1]
#     print(f"Predicted probability for sweet pepper: {proba:.4f}")
#     pred = int(proba > threshold)
#     print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")

def extract_features_hsv_hist_concat(X, h_bins=8, s_bins=4, v_bins=4):
    """
    Concatenates flattened HSV image and HSV histogram features for a batch of images.
    """
    feats = []
    for img in X:
        hsv = rgb_to_hsv_flat(img)
        hsv_flat = hsv.flatten()
        hist = extract_hsv_histogram_skimage(img, h_bins=h_bins, s_bins=s_bins, v_bins=v_bins)
        combined = np.concatenate([hsv_flat, hist])
        feats.append(combined)
    return np.array(feats)

def predict_gmm_margin(gmm_p, gmm_b, X, n_pepper, n_bg, N, margin=1.0):
    """
    Predicts class using two GMMs and a margin.
    """
    lp = gmm_p.score_samples(X) + np.log(n_pepper/N)
    lb = gmm_b.score_samples(X) + np.log(n_bg/N)
    return np.where(lp > lb + margin, 1, 0)

@timing
def process_input_image_gmm(image_path, scaler, gmm_p, gmm_b, n_pepper, n_bg, N, margin=1.0):
    """
    processes an input image, extracts HSV+hist features, scales, and predicts using GMMs.
    Returns predicted class (0 or 1).
    """
    image = imread(image_path)
    if image.max() > 1.0:
        image = image / 255.0
    image_resized = resize(image, (64, 64, 3), preserve_range=True, anti_aliasing=True)
    from libs.svm_utils import extract_features_hsv_hist_concat, predict_gmm_margin
    image_feat = extract_features_hsv_hist_concat([image_resized])
    image_feat = scaler.transform(image_feat)
    pred = predict_gmm_margin(gmm_p, gmm_b, image_feat, n_pepper, n_bg, N, margin=margin)[0]
    return pred