import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from libs.logreg_utils import plot_precision_recall_curve, plot_precision_recall_curve_save
from tqdm import tqdm
from libs.svm_utils import predict_gmm_margin

def save_model(model, model_name, model_data=None):
    """
    Save model to both trained and active directories
    """
    os.makedirs("models/trained", exist_ok=True)
    os.makedirs("models/active", exist_ok=True)
    
    trained_path = f"models/trained/{model_name}.pkl"
    if model_data is not None:
        with open(trained_path, 'wb') as f:
            pickle.dump(model_data, f)
    else:
        with open(trained_path, 'wb') as f:
            pickle.dump(model, f)
    
    active_path = f"models/active/{model_name}.pkl"
    if model_data is not None:
        with open(active_path, 'wb') as f:
            pickle.dump(model_data, f)
    else:
        with open(active_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Model saved to: {trained_path}")
    print(f"Model copied to active: {active_path}")

def load_model(model_name):
    """
    Load model from active directory
    """
    model_path = f"models/active/{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Please train the model first.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def print_metrics_and_plot(y_true, y_pred, y_scores, set_name, model_name, save_plots=False):
    """
    Print classification metrics and plot precision-recall curve
    """
    print(f"\n=== {set_name.upper()} SET METRICS ===")
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Pepper Class F1-Score: {f1:.4f}")
    print(f"Pepper Class Precision: {precision:.4f}")
    print(f"Pepper Class Recall: {recall:.4f}")
    
    # Calculate class distribution
    n_pepper = np.sum(y_true == 1)
    n_background = np.sum(y_true == 0)
    total = len(y_true)
    print(f"\nClass Distribution:")
    print(f"  Pepper pixels: {n_pepper} ({n_pepper/total:.1%})")
    print(f"  Background pixels: {n_background} ({n_background/total:.1%})")
    
    print(f"\nDetailed Classification Report:")
    print("  Class 0 = Background, Class 1 = Pepper")
    print(classification_report(y_true, y_pred, target_names=['Background', 'Pepper']))
    
    title = f"{model_name} - Precision-Recall Curve ({set_name} Set)"
    
    if save_plots:
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        # Create filename-safe model name
        safe_model_name = model_name.replace(" ", "_").replace("-", "_").lower()
        plot_filename = f"plots/{safe_model_name}_{set_name.lower()}_precision_recall.png"
        plot_precision_recall_curve_save(y_true, y_scores, title=title, save_path=plot_filename)
        print(f"Precision-recall curve saved to: {plot_filename}")
    else:
        plot_precision_recall_curve(y_true, y_scores, title=title)

# def get_predictions_and_scores(classifier, X_feat, has_predict_proba=True):
#     """
#     Get predictions and scores from a classifier, handling different types
#     """
#     if hasattr(classifier, 'predict_proba') and has_predict_proba:
#         y_pred = classifier.predict(X_feat)
#         y_scores = classifier.predict_proba(X_feat)[:, 1]
#     elif hasattr(classifier, 'predict_proba_custom'):
#         y_pred = classifier.predict(X_feat)
#         y_scores = classifier.predict_proba_custom(X_feat)[:, 1]
#     else:
#         y_pred = classifier.predict(X_feat)
#         decision_scores = classifier.decision_function(X_feat) if hasattr(classifier, 'decision_function') else y_pred
#         y_scores = decision_scores
#     
#     return y_pred, y_scores

def evaluate_gmm_model(X_feat, y_true, gmm_p, gmm_b, n_pepper, n_bg, N, margin=1.0):
    """
    Evaluate GMM model and return predictions and scores
    """
    y_pred = []
    y_scores = []
    
    for i in tqdm(range(len(X_feat)), desc="Processing samples"):
        pred = predict_gmm_margin(gmm_p, gmm_b, X_feat[i].reshape(1, -1), n_pepper, n_bg, N, margin)[0]
        y_pred.append(pred)
        
        #calculate scores for precision-recall curve
        lp = gmm_p.score_samples(X_feat[i].reshape(1, -1)) + np.log(n_pepper/N)
        lb = gmm_b.score_samples(X_feat[i].reshape(1, -1)) + np.log(n_bg/N)
        score = lp[0] - lb[0]  #difference as score
        y_scores.append(score)
    
    return np.array(y_pred), np.array(y_scores)
