import pickle
import numpy as np
from skimage.io import imsave
import os

def save_images_from_validation(pkl_path, label_name, out_dir, n=5):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    images = data['validation']
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images[:n]):
        # Ensure image is in [0,255] uint8
        img_to_save = img
        if img_to_save.max() <= 1.0:
            img_to_save = (img_to_save * 255).astype(np.uint8)
        else:
            img_to_save = img_to_save.astype(np.uint8)
        out_path = os.path.join(out_dir, f"{label_name}_val_{i+1}.png")
        imsave(out_path, img_to_save)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    # Save 5 background and 5 sweet pepper validation images
    save_images_from_validation('data/Q2_BG_dict.pkl', 'bg', 'validation_pngs', n=5)
    save_images_from_validation('data/Q2_SP_dict.pkl', 'sp', 'validation_pngs', n=5)
