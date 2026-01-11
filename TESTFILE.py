def extract_hsv_histogram_skimage(img, h_bins=8, s_bins=4, v_bins=4):
    """
    extracts a normalized HSV histogram from an image using skimage.
    args:
        img (np.ndarray): input image, shape (..., 3), values in [0,255] or [0,1].
        h_bins, s_bins, v_bins (int): number of bins for H, S, V channels.
    returns:
        np.ndarray: flattened and normalized histogram vector.
    """
    arr = np.array(img, dtype=np.float32)  # Convert input to float32 numpy array
    if arr.max() > 1.0:
        arr = arr / 255.0  # Normalize to [0,1] if needed
    hsv = color.rgb2hsv(arr)  # Convert RGB to HSV color space
    h = hsv[..., 0] * 180     # Scale H channel from [0,1] to [0,180] (OpenCV convention)
    s = hsv[..., 1] * 255     # Scale S channel from [0,1] to [0,255]
    v = hsv[..., 2] * 255     # Scale V channel from [0,1] to [0,255]
    # Compute a 3D histogram over H, S, V channels
    hist, _ = np.histogramdd(
        sample=[h.ravel(), s.ravel(), v.ravel()],  # Flatten each channel and stack as sample list
        bins=[h_bins, s_bins, v_bins],             # Number of bins for each channel
        range=[[0,180], [0,256], [0,256]]          # Value range for each channel
    )
    hist = hist.flatten()                          # Flatten the 3D histogram to 1D
    hist /= (hist.sum() + 1e-6)                    # Normalize histogram so sum is 1 (avoid division by zero)
    return hist