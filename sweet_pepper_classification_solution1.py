from libs.svm_utils import extract_features_hsv_lbp, process_input_imageQ2
from libs.data_utils import load_q2_data
from libs.model_utils import save_model, load_model, print_metrics_and_plot
import argparse
from sklearn.svm import SVC
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, nargs='?', help='path to input image')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--C', type=float, default=1.0, help='SVM regularization parameter')
parser.add_argument('--gamma', type=str, default='scale', help='SVM gamma parameter')
parser.add_argument('--kernel', type=str, default='rbf', help='SVM kernel type')
parser.add_argument('--model_name', type=str, default='sweet_pepper_classification_svm', help='Name for the model')
parser.add_argument('--save_plots', action='store_true', help='Save precision-recall curve plots to plots/ directory')
args = parser.parse_args()

def train_svm_model(C=1.0, gamma='scale', kernel='rbf', model_name="sweet_pepper_classification_svm", save_plots=False):
    """
    Train SVM model with specified hyperparameters
    """
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()
    
    # Feature extraction with progress bar
    print("Extracting features...")
    datasets = [("train", X_train), ("eval", X_eval), ("valid", X_valid)]
    features = {}
    
    for name, data in tqdm(datasets, desc="Feature extraction"):
        features[f"{name}_feat"] = extract_features_hsv_lbp(data)
    
    X_train_feat = features["train_feat"]
    X_eval_feat = features["eval_feat"]
    X_valid_feat = features["valid_feat"]
    
    print(f"Training SVM classifier with C={C}, gamma={gamma}, kernel={kernel}...")
    classifier = SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', probability=True)
    classifier.fit(X_train_feat, y_train)

    # Evaluation on evaluation set
    y_eval_pred = classifier.predict(X_eval_feat)
    y_eval_scores = classifier.predict_proba(X_eval_feat)[:, 1]
    print_metrics_and_plot(y_eval, y_eval_pred, y_eval_scores, "evaluation", "Sweet Pepper Classification SVM", save_plots=save_plots)

    # Evaluation on validation set
    y_valid_pred = classifier.predict(X_valid_feat)
    y_valid_scores = classifier.predict_proba(X_valid_feat)[:, 1]
    print_metrics_and_plot(y_valid, y_valid_pred, y_valid_scores, "validation", "Sweet Pepper Classification SVM", save_plots=save_plots)

    save_model(classifier, model_name)
    return classifier

def classify_with_svm(input_image, model_name="sweet_pepper_classification_svm"):
    """
    Classify image using trained SVM model
    """
    try:
        classifier = load_model(model_name)
        process_input_imageQ2(input_image, classifier)
    except FileNotFoundError:
        print("No trained model found. Training model with default parameters...")
        train_svm_model(model_name=model_name, save_plots=False)
        classify_with_svm(input_image, model_name)

def main():
    if args.train:
        train_svm_model(C=args.C, gamma=args.gamma, kernel=args.kernel, model_name=args.model_name, save_plots=args.save_plots)
    elif args.input_image:
        classify_with_svm(args.input_image, model_name=args.model_name)
    else:
        print("Please specify either --train or provide an input image path")

if __name__ == "__main__":
    main()

