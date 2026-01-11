from libs.logreg_utils import MyLogReg, plot_mask
from libs.data_utils import rgb_to_hsv_flat
import numpy as np

# Suppose images is a numpy array of shape (N, H, W, 3)
# For demonstration, let's say you have one image:
images = [...]  # list or array of RGB images

X_train = []
y_train = []

# Threshold parameters
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
    mask = hue_mask & sat_mask & val_mask  # This is your pseudo-label

    # Use hue and saturation as features
    hs = np.stack([hsv_img[..., 0], hsv_img[..., 1]], axis=-1)
    X_train.append(hs.reshape(-1, 2))
    y_train.append(mask.flatten().astype(int))

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)

# Train logistic regression on pseudo-labels
model = MyLogReg(learning_rate=0.0001, num_iterations=100)
model.fit(X_train, y_train)
