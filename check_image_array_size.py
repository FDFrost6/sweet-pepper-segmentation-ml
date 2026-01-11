import argparse
from skimage.io import imread
import numpy as np
from libs.knn_utils_test import convert_color_space, manual_normalize

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to input image file')
parser.add_argument('--color_space', choices=['rgb', 'lab'], default='rgb', help='Color space to use for features')
args = parser.parse_args()

image = imread(args.image)
if image.max() > 1.0:
    image = image / 255.0
pixels = image.reshape(-1, 3)
pixels = manual_normalize(convert_color_space(pixels, args.color_space))

print(f"Transformed image array shape: {pixels.shape}")
print(f"Data type: {pixels.dtype}")
print(f"Total size: {pixels.nbytes/1024/1024:.2f} MB")
