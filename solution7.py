from libs.svm_utils import extract_features_hsv_lbp, process_input_imageQ2
from libs.data_utils import load_q2_data
from libs.model_utils import save_model, load_model, print_metrics_and_plot
import argparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, nargs='?', help='path to input image')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--n_components', type=int, default=20, help='Number of PCA components')
parser.add_argument('--classifier', type=str, default='hist_gb', choices=['hist_gb', 'random_forest'], help='Type of classifier to use')
parser.add_argument('--model_name', type=str, default='sweet_pepper_classification_ensemble', help='Name for the model')
args = parser.parse_args()

def train_gradient_boosting_model(n_components=20, classifier_type='hist_gb', model_name="sweet_pepper_classification_ensemble"):
    """
    Train Gradient Boosting model with PCA and specified hyperparameters
    """
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = load_q2_data()

    # Feature extraction with progress bar
    print("Extracting features...")
    datasets = [("train", X_train), ("valid", X_valid)]
    features = {}
    
    for name, data in tqdm(datasets, desc="Feature extraction"):
        features[f"{name}_feat"] = extract_features_hsv_lbp(data)
    
    X_train_feat = features["train_feat"]
    X_valid_feat = features["valid_feat"]

    print(f"Training {classifier_type} classifier with PCA n_components={n_components}...")
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    
    if classifier_type == 'hist_gb':
        classifier = HistGradientBoostingClassifier(random_state=42)
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    else:
        raise ValueError("classifier_type must be 'hist_gb' or 'random_forest'")
    
    pipeline = Pipeline([('pca', PCA(n_components=n_components, random_state=42)), ('clf', classifier)])
    
    if classifier_type == 'hist_gb':
        pipeline.fit(X_train_feat, y_train, clf__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train_feat, y_train)

    y_valid_pred = pipeline.predict(X_valid_feat)
    y_valid_scores = pipeline.predict_proba(X_valid_feat)[:, 1]
    print_metrics_and_plot(y_valid, y_valid_pred, y_valid_scores, "validation", f"{classifier_type.upper()} Solution7")

    save_model(pipeline, model_name)
    return pipeline

def classify_with_gradient_boosting(input_image, model_name="sweet_pepper_classification_ensemble"):
    """
    Classify image using trained gradient boosting model
    """
    try:
        pipeline = load_model(model_name)
        process_input_imageQ2(input_image, pipeline)
    except FileNotFoundError:
        print("No trained model found. Training model with default parameters...")
        train_gradient_boosting_model(model_name=model_name)
        classify_with_gradient_boosting(input_image, model_name)

def main():
    if args.train:
        train_gradient_boosting_model(n_components=args.n_components, classifier_type=args.classifier, model_name=args.model_name)
    elif args.input_image:
        classify_with_gradient_boosting(args.input_image, model_name=args.model_name)
    else:
        print("Please specify either --train or provide an input image path")

if __name__ == "__main__":
    main()
