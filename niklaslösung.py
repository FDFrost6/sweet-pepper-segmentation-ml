import argparse
import matplotlib.pyplot as plt

from skimage.io import imread

from libs.Models_and_Functions import LogisticRegression, PepperSegment, segment_image, eval_and_plot
from libs.loading_data import data_loader_Q1

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--lr",   type=float, default=0.1, help="Learning rate")
parser.add_argument("--iter", type=int,   default=500,   help="Max iterations")
flags = parser.parse_args()


def main():
    # Data loading
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_Q1()

    # Modells and Segmenter, setting flags for model
    seg_rgb = PepperSegment(LogisticRegression(learningrate=flags.lr, max_iteration=flags.iter),
                            color_space="rgb", threshhold=0.7)
    seg_lab = PepperSegment(LogisticRegression(learningrate=flags.lr, max_iteration=flags.iter),
                            color_space="lab", threshhold=1)

    # 3) Training
    seg_rgb.fitting(X_train, y_train)
    seg_lab.fitting(X_train, y_train)

    # 4) Evaluation
    eval_and_plot("RGB - Eval", seg_rgb, X_eval, y_eval)
    eval_and_plot("Lab - Eval", seg_lab, X_eval, y_eval)

    # 5) Validation
    eval_and_plot("RGB - Valid", seg_rgb, X_valid, y_valid)
    eval_and_plot("Lab - Valid", seg_lab, X_valid, y_valid)

    # 6) Image
    img = imread(flags.image)
    mask_rgb = segment_image(seg_rgb, img)
    mask_lab = segment_image(seg_lab, img)

    for name, mask in [("RGB", mask_rgb), ("Lab", mask_lab)]:
        pct = mask.sum() / mask.size * 100
        print(f"{name}: Paprika-Pixel = {pct:.2f}%")

    # 7) Plotting the image with mask
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title("Original"); plt.axis("off"); plt.imshow(img)
    plt.subplot(1,2,2); plt.title("Mask RGB"); plt.axis("off"); plt.imshow(mask_rgb, cmap="gray")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title("Original"); plt.axis("off"); plt.imshow(img)
    plt.subplot(1,2,2); plt.title("Mask Lab"); plt.axis("off"); plt.imshow(mask_lab, cmap="gray")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()