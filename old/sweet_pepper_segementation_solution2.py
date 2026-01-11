import argparse
from libs.data_utils import PepperDataLoader
from libs.knn_utils import (
    train_knn,
    predict_with_progress,
    plot_knn_data_distribution,
    plot_precision_recall_standard,
    print_classification_report,
    time_function,
    convert_color_space,
    normalize_features
)

# Example usage:
# python sweet_pepper_segementation_solution1.py --files Q1_BG_dict.pkl Q1_Red_dict.pkl Q1_Yellow_dict.pkl --labels 0 1 2 --data_type vector --n_neighbors 5 --color_space lab --show_plots

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True, help='List of pickle files')
parser.add_argument('--labels', nargs='+', type=int, required=True, help='List of class labels')
parser.add_argument('--data_type', choices=['vector', 'image'], default='vector', help='Type of data')
parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
parser.add_argument('--color_space', choices=['rgb', 'lab'], default='rgb', help='Color space to use for features')
parser.add_argument('--show_plots', action='store_true', help='Show precision-recall and data distribution plots')
parser.add_argument('--plot_distribution', action='store_true', help='Plot KNN data distribution (2D)')
args = parser.parse_args()

# Build the mapping from files to labels
file_label_map = dict(zip(args.files, args.labels))

# Initialize and build dataset
loader = PepperDataLoader(file_label_map, data_type=args.data_type)
loader.build_dataset()
X_train, y_train = loader.get_set('train')
X_val, y_val = loader.get_set('validation')
X_eval, y_eval = loader.get_set('evaluation')

# Convert color space and normalize
X_train = normalize_features(convert_color_space(X_train, args.color_space))
X_val = normalize_features(convert_color_space(X_val, args.color_space))
X_eval = normalize_features(convert_color_space(X_eval, args.color_space))

# Train KNN and time it
knn = train_knn(X_train, y_train, n_neighbors=args.n_neighbors)

# Validation set prediction with progress bar
y_val_pred, y_val_proba = predict_with_progress(knn, X_val, desc="Validation Prediction")

# Evaluation set prediction with progress_bar
y_eval_pred, y_eval_proba = predict_with_progress(knn, X_eval, desc="Evaluation Prediction")

# Plots
if args.show_plots:
    if args.plot_distribution:
        plot_knn_data_distribution(X_train, y_train, title=f"KNN Training Data Distribution ({args.color_space.upper()})")
    class_names = [str(lbl) for lbl in args.labels]
    plot_precision_recall_standard(y_val, y_val_proba, class_names=class_names, title="Validation Precision-Recall (One-vs-Rest)")
    plot_precision_recall_standard(y_eval, y_eval_proba, class_names=class_names, title="Evaluation Precision-Recall (One-vs-Rest)")

# Print classification reports for validation and evaluation sets
print("Validation Classification Report:")
print_classification_report(y_val, y_val_pred, target_names=class_names)
print("Evaluation Classification Report:")
print_classification_report(y_eval, y_eval_pred, target_names=class_names)


