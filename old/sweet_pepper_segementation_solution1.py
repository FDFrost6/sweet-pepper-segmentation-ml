import argparse
from libs.data_utils import PepperDataLoader
from libs.knn_utils import (
    train_custom_knn,
    predict_custom_knn,
    plot_knn_data_distribution,
    print_classification_report,
    convert_color_space,
    manual_normalize,
    calc_batch_size
)

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True, help='List of pickle files')
parser.add_argument('--labels', nargs='+', type=int, required=True, help='List of class labels')
parser.add_argument('--data_type', choices=['vector', 'image'], default='vector', help='Type of data')
parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
parser.add_argument('--color_space', choices=['rgb', 'lab'], default='rgb', help='Color space to use for features')
parser.add_argument('--show_plots', action='store_true', help='Show precision-recall and data distribution plots')
parser.add_argument('--plot_distribution', action='store_true', help='Plot KNN data distribution (2D)')
parser.add_argument('--ram_gb', type=int, default=5, help='RAM (in GB) to allocate for KNN batching (default: 5GB)')
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
X_train = manual_normalize(convert_color_space(X_train, args.color_space))
X_val = manual_normalize(convert_color_space(X_val, args.color_space))
X_eval = manual_normalize(convert_color_space(X_eval, args.color_space))

# Train KNN and time it
knn = train_custom_knn(X_train, y_train, n_neighbors=args.n_neighbors)


# Calculate batch size for user-specified RAM
n_train = X_train.shape[0]
n_features = X_train.shape[1]
batch_size = calc_batch_size(args.ram_gb, n_train, n_features)

# Validation set prediction
y_val_pred = predict_custom_knn(knn, X_val, desc="Validation Prediction", batch_size=batch_size)

# Evaluation set prediction
y_eval_pred = predict_custom_knn(knn, X_eval, desc="Evaluation Prediction", batch_size=batch_size)

# Plots
if args.show_plots:
    if args.plot_distribution:
        plot_knn_data_distribution(X_train, y_train, title=f"Custom KNN Training Data Distribution ({args.color_space.upper()})")
    class_names = [str(lbl) for lbl in args.labels]
    # No probability output for custom KNN, so skip precision-recall curves

# Print classification reports for validation and evaluation sets
print("Validation Classification Report:")
print_classification_report(y_val, y_val_pred, target_names=class_names)
print("Evaluation Classification Report:")
print_classification_report(y_eval, y_eval_pred, target_names=class_names)

