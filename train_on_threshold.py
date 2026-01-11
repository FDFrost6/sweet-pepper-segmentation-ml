from libs.logreg_utils import MyLogReg, plot_mask
from libs.data_utils import rgb_to_hsv_flat
from skimage.io import imread
import numpy as np

def normalize_rgb_to_hue_sat(input_array):
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

# Load your training images (as a list or array)
images = [...]  # Replace with your list of RGB images

X_train = []
y_train = []

hmin, hmax, smin, smax = (0.95, 0.05, 0.4, 1.0)
vmin, vmax = (0.2, 1.0)

for img in images:
    hsv_img = rgb_to_hsv_flat(img)
    if hmin > hmax:
        hue_mask = (hsv_img[..., 0] >= hmin) | (hsv_img[..., 0] <= hmax)
    else:
        hue_mask = (hsv_img[..., 0] >= hmin) & (hsv_img[..., 0] <= hmax)
    sat_mask = (hsv_img[..., 1] >= smin) & (hsv_img[..., 1] <= smax)
    val_mask = (hsv_img[..., 2] >= vmin) & (hsv_img[..., 2] <= vmax)
    mask = hue_mask & sat_mask & val_mask  # pseudo-labels

    hs = np.stack([hsv_img[..., 0], hsv_img[..., 1]], axis=-1)
    X_train.append(hs.reshape(-1, 2))
    y_train.append(mask.flatten().astype(int))

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

# Train logistic regression on pseudo-labels
model = MyLogReg(learning_rate=0.001, num_iterations=100)
model.fit(X_train, y_train)
