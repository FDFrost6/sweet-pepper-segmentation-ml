from libs.logreg_utils import MyLogReg, plot_segmentation_mask
from libs.data_utils import data_loader_q1, normalize_rgb_to_lab, process_input_image
from libs.model_utils import save_model, load_model, print_metrics_and_plot
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, nargs='?', help='path to input image to segment')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for logistic regression')
parser.add_argument('--num_iterations', type=int, default=50, help='Number of iterations for training')
parser.add_argument('--model_name', type=str, default='sweet_pepper_segmentation_logreg', help='Name for the model')
parser.add_argument('--show_mask', action='store_true', help='Show segmentation mask visualization')
parser.add_argument('--save_plots', action='store_true', help='Save precision-recall curve plots to plots/ directory')
args = parser.parse_args()

def train_logreg_model(learning_rate=0.0001, num_iterations=50, model_name="sweet_pepper_segmentation_logreg", save_plots=False):
    """
    Train custom logistic regression model with specified hyperparameters
    """
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()
    
    X_train_lab = normalize_rgb_to_lab(X_train)
    X_eval_lab = normalize_rgb_to_lab(X_eval)
    X_valid_lab = normalize_rgb_to_lab(X_valid)

    print(f"Training model with learning_rate={learning_rate}, num_iterations={num_iterations}")
    classifier = MyLogReg(learning_rate=learning_rate, num_iterations=num_iterations)
    classifier.fit(X_train_lab, y_train)

    # Evaluation on evaluation set
    y_eval_pred = classifier.predict(X_eval_lab)
    y_eval_scores = classifier.predict_proba(X_eval_lab)[:, 1]
    print_metrics_and_plot(y_eval, y_eval_pred, y_eval_scores, "evaluation", "Sweet Pepper Segmentation LogReg", save_plots=save_plots)

    # Evaluation on validation set
    y_valid_pred = classifier.predict(X_valid_lab)
    y_valid_scores = classifier.predict_proba(X_valid_lab)[:, 1]
    print_metrics_and_plot(y_valid, y_valid_pred, y_valid_scores, "validation", "Sweet Pepper Segmentation LogReg", save_plots=save_plots)

    save_model(classifier, model_name)
    return classifier

def classify_with_logreg(input_image, model_name="sweet_pepper_segmentation_logreg", show_mask=False):
    """
    Classify image using trained logistic regression model
    """
    try:
        classifier = load_model(model_name)
        y_pred_image, orig_shape, image = process_input_image(input_image, classifier, normalize_rgb_to_lab)
        percent_pepper = 100.0 * np.sum(y_pred_image == 1) / y_pred_image.size
        
        if show_mask:
            mask = y_pred_image.reshape(orig_shape)
            plot_segmentation_mask(mask, original_image=image)
        
        print(f"identified: {percent_pepper:.2f}% of total pixels as sweetpepper")
        return percent_pepper
    except FileNotFoundError:
        print("No trained model found. Training model with default parameters...")
        train_logreg_model(model_name=model_name, save_plots=False)
        return classify_with_logreg(input_image, model_name, show_mask)

def main():
    if args.train:
        train_logreg_model(learning_rate=args.learning_rate, num_iterations=args.num_iterations, model_name=args.model_name, save_plots=args.save_plots)
    elif args.input_image:
        classify_with_logreg(args.input_image, model_name=args.model_name, show_mask=args.show_mask)
    else:
        print("Please specify either --train or provide an input image path")

if __name__ == "__main__":
    main()