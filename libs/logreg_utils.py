import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from functools import wraps
import time
from tqdm import tqdm


class MyLogReg:
    """
    binary logistic regression using gradient descent.
    this class supports sample weighting for imbalanced datasets and provides methods for fitting
    predicting probabilities, and predicting class labels.
    """
    def __init__(self, learning_rate, num_iterations):
        """
        initialize the logistic regression model.
        args:learning_rate: step size for gradient descent updates.
             num_iterations: number of iterations for training.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None  # model weights (coefficients)
        self.bias = None     # model bias (intercept)

    def sigmoid(self, input_value):
        """
        compute the sigmoid activation function.
        args: input_value: linear combination of features and weights.
        returns: np.ndarray: output after applying sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-input_value))

    def fit(self, feature_matrix, target_vector):
        """
        train the logistic regression model using gradient descent.
        args:feature_matrix: training data of shape (num_samples, num_features).
             target_vector: binary target labels of shape (num_samples).
        """
        num_samples, num_features = feature_matrix.shape
        self.weights = np.zeros(num_features, dtype=float)
        self.bias = 0.0

        #compute class sample weights for handling class imbalance
        num_class0 = np.sum(target_vector == 0)
        num_class1 = np.sum(target_vector == 1)
        sample_weight_class0 = num_samples / (2 * num_class0)
        sample_weight_class1 = num_samples / (2 * num_class1)
        sample_weights = np.where(target_vector == 1, sample_weight_class1, sample_weight_class0)

        #gradient descent optimization
        for i in tqdm(range(self.num_iterations), desc="Training LogReg", unit="iteration"):
            #linear combination of features and weights
            linear_output = feature_matrix @ self.weights + self.bias
            #predicted probabilities for class 1
            predicted_probabilities = self.sigmoid(linear_output)
            #compute weighted gradient
            gradient = (predicted_probabilities - target_vector) * sample_weights
            gradient_weights = (feature_matrix.T @ gradient) / num_samples
            gradient_bias = np.sum(gradient) / num_samples
            #update weights and bias
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict_proba(self, feature_matrix):
        """
        predict class probabilities for input samples.
        args: feature_matrix: input data of shape (num_samples, num_features).
        returns: array of shape (num_samples, 2) with probabilities for classes 0 and 1.
        """
        linear_combination = feature_matrix @ self.weights + self.bias
        probability_class1 = self.sigmoid(linear_combination)
        probability_class0 = 1.0 - probability_class1
        return np.column_stack([probability_class0, probability_class1])

    def predict(self, feature_matrix):
        """
        predict binary class labels for input samples.
        args:feature_matrix: input data of shape (num_samples, num_features).
        returns: predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(feature_matrix)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        return predictions
    
def plot_precision_recall_curve(y_true, y_scores, title="precision-recall curve"):
    """
    plots the precision-recall curve for binary classification.
    Shows curve for positive class (pepper = class 1)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'Pepper Class (AP={avg_precision:.3f})', linewidth=2)
    plt.xlabel('Recall (Pepper Class)', fontsize=12)
    plt.ylabel('Precision (Pepper Class)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()

def plot_precision_recall_curve_save(y_true, y_scores, title="precision-recall curve", save_path="precision_recall.png"):
    """
    plots and saves the precision-recall curve for binary classification.
    Shows curve for positive class (pepper = class 1)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'Pepper Class (AP={avg_precision:.3f})', linewidth=2)
    plt.xlabel('Recall (Pepper Class)', fontsize=12)
    plt.ylabel('Precision (Pepper Class)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

def timing(func):
    """
    decorator to print the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"It took {end - start:.2f} seconds to process your input image")
        return result
    return wrapper

def plot_segmentation_mask(mask, original_image=None, title="segmentation mask"):
    """
    overlays the segmentation mask (pepper pixels in black) on the original image.
    """
    img = original_image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    overlay = img.copy()
    overlay[mask == 1] = [0, 0, 0]
    plt.figure(figsize=(6, 8))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    plt.show()

