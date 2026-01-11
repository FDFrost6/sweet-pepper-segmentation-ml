import pickle
import numpy as np
from skimage import color
from skimage.io import imread
from libs.logreg_utils import timing

def data_loader_q1():
    """
    loads Q1_BG_dict.pkl and Q1_Red_dict.pkl and returns stacked train/eval/valid sets with labels 0 (bg) and 1 (red).
    returns: x_train, y_train, x_eval, y_eval, x_valid, y_valid
    """
    with open('Q1_BG_dict.pkl', 'rb') as f:
        bg = pickle.load(f)
    with open('Q1_Red_dict.pkl', 'rb') as f:
        red = pickle.load(f)
    x_train = np.concatenate([np.stack(bg['train']), np.stack(red['train'])], axis=0)
    y_train = np.concatenate([np.zeros(len(bg['train'])), np.ones(len(red['train']))])
    x_eval = np.concatenate([np.stack(bg['evaluation']), np.stack(red['evaluation'])], axis=0)
    y_eval = np.concatenate([np.zeros(len(bg['evaluation'])), np.ones(len(red['evaluation']))])
    x_valid = np.concatenate([np.stack(bg['validation']), np.stack(red['validation'])], axis=0)
    y_valid = np.concatenate([np.zeros(len(bg['validation'])), np.ones(len(red['validation']))])
    return x_train, y_train, x_eval, y_eval, x_valid, y_valid

def data_loader_q1_extra():
    """
    loads Q1_BG_dict.pkl, Q1_Red_dict.pkl, Q1_Yellow_dict.pkl and returns stacked train/eval/valid sets with labels 0 (bg), 1 (red), 2 (yellow).
    returns: x_train, y_train, x_eval, y_eval, x_valid, y_valid
    """
    with open('Q1_BG_dict.pkl', 'rb') as f:
        bg = pickle.load(f)
    with open('Q1_Red_dict.pkl', 'rb') as f:
        red = pickle.load(f)
    with open('Q1_Yellow_dict.pkl', 'rb') as f:
        yellow = pickle.load(f)
    x_train = np.concatenate([np.stack(bg['train']), np.stack(red['train']), np.stack(yellow['train'])], axis=0)
    y_train = np.concatenate([np.zeros(len(bg['train'])), np.ones(len(red['train'])), np.full(len(yellow['train']), 2)])
    x_eval = np.concatenate([np.stack(bg['evaluation']), np.stack(red['evaluation']), np.stack(yellow['evaluation'])], axis=0)
    y_eval = np.concatenate([np.zeros(len(bg['evaluation'])), np.ones(len(red['evaluation'])), np.full(len(yellow['evaluation']), 2)])
    x_valid = np.concatenate([np.stack(bg['validation']), np.stack(red['validation']), np.stack(yellow['validation'])], axis=0)
    y_valid = np.concatenate([np.zeros(len(bg['validation'])), np.ones(len(red['validation'])), np.full(len(yellow['validation']), 2)])
    return x_train, y_train, x_eval, y_eval, x_valid, y_valid


def normalize_rgb_to_lab(input_array):
    """
    normalizes rgb input array to [0,1] and converts it to lab color space.
    args:input_array of shape (..., 3) with rgb values in [0,255] or [0,1].
    returns:array of same shape (..., 3) in lab color space.
    """
    arr = np.array(input_array, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    # reshape for skimage if needed
    original_shape = arr.shape
    arr_reshaped = arr.reshape(-1, 1, 3)
    lab = color.rgb2lab(arr_reshaped)
    return lab.reshape(original_shape)

def rgb_to_hsv_flat(image):
    """
    converts rgb image to hsv color space and returns the full hsv image.
    args:rgb image, shape (..., 3), values in [0,255] or [0,1].
    returns: hsv image, same shape as input.
    """
    arr = np.array(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    orig_shape = arr.shape
    arr_reshaped = arr.reshape(-1, 1, 3)
    hsv = color.rgb2hsv(arr_reshaped)
    return hsv.reshape(orig_shape)

@timing
def process_input_image(image_path, model, normalize_func):
    """
    loads an image, normalizes and reshapes it, and predicts pixel classes using the provided classifier.
    returns the predicted labels, the original image shape, and the image itself.
    """
    image = imread(image_path)
    orig_shape = image.shape[:2]
    image_lab = normalize_func(image.reshape(-1, 3))
    y_pred_img = model.predict(image_lab)
    return y_pred_img, orig_shape, image

def load_q2_data():
    """
    loads Q2_BG_dict.pkl and Q2_SP_dict.pkl and returns stacked train/eval/valid sets with labels 0 (bg) and 1 (sweet pepper).
    returns: X_train, y_train, X_eval, y_eval, X_valid, y_valid
    """
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