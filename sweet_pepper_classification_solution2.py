from libs.data_utils import load_q2_data
from libs.svm_utils import extract_features_hsv_hist_concat, predict_gmm_margin, process_input_image_gmm
from libs.model_utils import save_model, load_model, print_metrics_and_plot, evaluate_gmm_model
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, nargs='?', help='path to input image')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--n_components_pepper', type=int, default=3, help='Number of components for pepper GMM')
parser.add_argument('--n_components_bg', type=int, default=3, help='Number of components for background GMM')
parser.add_argument('--covariance_type', type=str, default='diag', help='Covariance type for GMM')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for GMM prediction')
parser.add_argument('--model_name', type=str, default='sweet_pepper_classification_gmm', help='Name for the model')
parser.add_argument('--save_plots', action='store_true', help='Save precision-recall curve plots to plots/ directory')
args = parser.parse_args()


def train_gmm_model(n_components_p=3, n_components_b=3, covariance_type="diag", margin=1.0, model_name="sweet_pepper_classification_gmm", save_plots=False):
    """
    Train GMM model with specified hyperparameters
    """
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()
    
    # Feature extraction with progress bar
    print("Extracting features...")
    datasets = [("train", X_train), ("eval", X_eval), ("valid", X_valid)]
    features = {}
    
    for name, data in tqdm(datasets, desc="Feature extraction"):
        features[f"{name}_feat"] = extract_features_hsv_hist_concat(data)
    
    X_train_feat = features["train_feat"]
    X_eval_feat = features["eval_feat"] 
    X_valid_feat = features["valid_feat"]
    
    # Feature scaling with progress bar
    print("Scaling features...")
    scaler = StandardScaler()
    scaling_steps = [("train", X_train_feat), ("eval", X_eval_feat), ("valid", X_valid_feat)]
    scaled_features = {}
    
    for i, (name, data) in enumerate(tqdm(scaling_steps, desc="Scaling features")):
        if name == "train":
            scaled_features[name] = scaler.fit_transform(data)
        else:
            scaled_features[name] = scaler.transform(data)
    
    X_train_feat = scaled_features["train"]
    X_eval_feat = scaled_features["eval"]
    X_valid_feat = scaled_features["valid"]
    
    #separate classes
    X_pepper = X_train_feat[y_train == 1]
    X_bg = X_train_feat[y_train == 0]
    feature_number = len(X_train_feat[0])

    print(f"Training GMM models with n_components_pepper={n_components_p}, n_components_bg={n_components_b}...")
    
    # Training GMMs with progress bar
    gmm_models = [("pepper", X_pepper, n_components_p), ("background", X_bg, n_components_b)]
    trained_gmms = {}
    
    for name, data, n_comp in tqdm(gmm_models, desc="Training GMMs"):
        gmm = GaussianMixture(n_components=n_comp, covariance_type=covariance_type, random_state=42)
        gmm.fit(data)
        trained_gmms[name] = gmm
    
    gmm_p = trained_gmms["pepper"]
    gmm_b = trained_gmms["background"]

    # Evaluation on evaluation set
    y_eval_pred, y_eval_scores = evaluate_gmm_model(X_eval_feat, y_eval, gmm_p, gmm_b, len(X_pepper), len(X_bg), len(X_train_feat), margin)
    print_metrics_and_plot(y_eval, y_eval_pred, y_eval_scores, "evaluation", "Sweet Pepper Classification GMM", save_plots=save_plots)

    # Evaluation on validation set
    y_valid_pred, y_valid_scores = evaluate_gmm_model(X_valid_feat, y_valid, gmm_p, gmm_b, len(X_pepper), len(X_bg), len(X_train_feat), margin)
    print_metrics_and_plot(y_valid, y_valid_pred, y_valid_scores, "validation", "Sweet Pepper Classification GMM", save_plots=save_plots)

    #save trained models and scaler
    model_data = {'scaler': scaler, 'gmm_p': gmm_p,'gmm_b': gmm_b, 'n_pepper': len(X_pepper), 'n_bg': len(X_bg),'feature_number': feature_number,'margin': margin}
    save_model(None, model_name, model_data)
    return model_data

def classify_with_gmm(input_image, model_name="sweet_pepper_classification_gmm"):
    """
    Classify image using trained GMM model
    """
    try:
        model_data = load_model(model_name)
        pred = process_input_image_gmm(input_image, model_data['scaler'], model_data['gmm_p'], model_data['gmm_b'], model_data['n_pepper'], model_data['n_bg'], model_data['feature_number'], margin=model_data['margin'])
        print("Prediction:", "sweet pepper" if pred == 1 else "not sweet pepper")
        return pred
    except FileNotFoundError:
        print("No trained model found. Training model with default parameters...")
        train_gmm_model(model_name=model_name, save_plots=False)
        return classify_with_gmm(input_image, model_name)
    
def main():
    if args.train:
        train_gmm_model(n_components_p=args.n_components_pepper, n_components_b=args.n_components_bg, covariance_type=args.covariance_type, margin=args.margin, model_name=args.model_name, save_plots=args.save_plots)
    elif args.input_image:
        classify_with_gmm(args.input_image, model_name=args.model_name)
    else:
        print("Please specify either --train or provide an input image path")

if __name__ == "__main__":
    main()

