# Sweet Pepper Segmentation and Classification

## Project Structure


```
Assignment/
├── libs/                           # Utility modules
│   ├── data_utils.py              # Data loading and preprocessing utilities
│   ├── logreg_utils.py            # Custom logistic regression and visualization
│   ├── svm_utils.py               # Feature extraction and SVM utilities
│   └── model_utils.py             # Model saving/loading and evaluation
├── models/                        # Model storage
│   ├── trained/                   # Newly trained models
│   └── active/                    # Models ready for use
├── plots/                         # Saved precision-recall curve plots
├── data/                          # Dataset files
├── sweet_pepper_segementation_solution1.py    # Custom Logistic Regression (LAB color space)
├── sweet_pepper_segementation_solution2.py    # K-Nearest Neighbors (LAB color space)  
├── sweet_pepper_segementation_solution_extra.py  # Multi-class KNN (3 classes)
├── sweet_pepper_classification_solution1.py   # SVM with HSV+LBP features
├── sweet_pepper_classification_solution2.py   # Gaussian Mixture Models
├── solution7.py                   # Gradient Boosting with PCA
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Solutions Overview

### Solution 1: Custom Logistic Regression
- **Algorithm**: Custom implementation with gradient descent (performed grid search to find optimal hyper parameters here)
- **Features**: LAB color space pixel values (LAB performed well here compared to RGB!)
- **Use Case**: Binary classification (pepper vs background)
- **Key Features**: Class balancing, custom training loop

### Solution 2: K-Nearest Neighbors
- **Algorithm**: KNN classifier (k=3 beat k=5 beause the training time was reduced while having diffrence of 0.01 in precision and recall for sweet pepper)
- **Features**: LAB color space pixel values (Same as before LAB was the my got to here)
- **Use Case**: Binary classification (pepper vs background)
- **Key Features**: Distance-based classification, precision-recall analysis

### Solution Extra: Multi-class KNN
- **Algorithm**: KNN classifier (k=3 beat k=5 beause the training time was reduced and the segmentation even looked better:)! 
- **Features**: LAB color space pixel values (Same as before LAB was the my got to here)
- **Use Case**: 3-class classification (background, red pepper, yellow pepper)
- **Key Features**: Multi-class segmentation, percentage analysis

### Classification Solution 1: Support Vector Machine
- **Algorithm**: SVM with RBF kernel (performed grid search for optimal parameters here)
- **Features**: HSV histograms + LBP (Local Binary Patterns) (Here HSV perfomed best compared with LAB and RGB)
- **Use Case**: Binary classification with advanced features
- **Key Features**: Feature engineering, probability estimation

### Classification Solution 2: Gaussian Mixture Models
- **Algorithm**: Dual GMM (pepper and background models) (tried diffrent features as input here but this was the most efficient combination in my case)
- **Features**: HSV histograms + flattened HSV values  (Here HSV also perfomed better then LAB or RGB)
- **Use Case**: Probabilistic classification with margin-based decisions
- **Key Features**: Unsupervised learning components, feature scaling

### Solution 7: Gradient Boosting
- **Algorithm**: Histogram Gradient Boosting or Random Forest (I was suprised how well out of the box HistGradBoost with PCA worked without any tuning, randomforest did not perfom well out of the box in comparison)
- **Features**: HSV+LBP features with PCA dimensionality reduction
- **Use Case**: Ensemble methods for robust classification
- **Key Features**: PCA preprocessing, ensemble learning

## Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
### Quick Start
```bash
# Test a solution with automatic model training
python sweet_pepper_segementation_solution1.py peppers.png
```
## Usage

### Basic Classification
All solutions support direct image classification:

```bash
# Solution 1: Custom Logistic Regression
python sweet_pepper_segementation_solution1.py peppers.png

# Solution 2: KNN Classification  
python sweet_pepper_segementation_solution2.py peppers.png

# Solution Extra: Multi-class KNN
python sweet_pepper_segementation_solution_extra.py peppers.png

# Classification Solution 1: SVM Classification
python sweet_pepper_classification_solution1.py sp_val_3.png

# Classification Solution 2: GMM Classification
python sweet_pepper_classification_solution2.py sp_val_3.png

# Solution 7: Gradient Boosting/Randomforest -> please dont grade this it was just for playing around with other models :)!
python solution7.py sp_val_3.png
```

### Visualization Options
Add visualization flags for detailed analysis:

```bash
# Show segmentation mask overlay
python sweet_pepper_segementation_solution1.py peppers.png --show_mask

# Show validation metrics and precision-recall curves  
python sweet_pepper_segementation_solution2.py peppers.png --show_metrics

# Combine both visualizations
python sweet_pepper_segementation_solution_extra.py peppers.png --show_mask --show_metrics

# Save precision-recall curves to plots/ directory (Segmentation Solution 1, Classification Solutions 1 & 2)
python sweet_pepper_segementation_solution1.py --train --save_plots
python sweet_pepper_classification_solution1.py --train --save_plots  
python sweet_pepper_classification_solution2.py --train --save_plots
```

### Advanced Options
```bash
# Custom model names for multiple experiments
python sweet_pepper_classification_solution1.py --train --model_name svm_experiment_1 --C 5.0
python sweet_pepper_classification_solution1.py sp_val_3.png --model_name svm_experiment_1

# Training with progress bars (Classification Solution 2)
python sweet_pepper_classification_solution2.py --train  # Shows tqdm progress for GMM evaluation
```

### Training Models
Train models with custom hyperparameters:

```bash
# Train custom logistic regression
python sweet_pepper_segementation_solution1.py --train --learning_rate 0.001 --num_iterations 100

# Train custom logistic regression with plot saving
python sweet_pepper_segementation_solution1.py --train --learning_rate 0.001 --num_iterations 100 --save_plots

# Train KNN with different neighbors
python sweet_pepper_segementation_solution2.py --train --n_neighbors 7

# Train SVM with custom parameters
python sweet_pepper_classification_solution1.py --train --C 10.0 --gamma 0.1 --kernel rbf

# Train SVM with plot saving
python sweet_pepper_classification_solution1.py --train --C 10.0 --gamma 0.1 --kernel rbf --save_plots

# Train GMM with custom components
python sweet_pepper_classification_solution2.py --train --n_components_pepper 5 --n_components_bg 3 --margin 2.0

# Train GMM with plot saving
python sweet_pepper_classification_solution2.py --train --n_components_pepper 5 --n_components_bg 3 --margin 2.0 --save_plots

# Train Gradient Boosting with PCA
python solution7.py --train --n_components 30 --classifier random_forest
```

## Model Storage

- **Trained models** are saved in `models/trained/` with timestamps
- **Active models** are copied to `models/active/` for immediate use
- **Plot outputs** are saved in `plots/` directory when using `--save_plots` flag
- Models are automatically loaded if available, or trained with defaults if missing

## Plot Saving Feature

Segmentation Solution 1 and Classification Solutions 1 & 2 support saving precision-recall curves to the `plots/` directory:

- **Flag**: `--save_plots` (optional, default: disabled)
- **Output**: PNG files with descriptive names (e.g., "Sweet_Pepper_Segmentation_LogReg_validation_precision_recall.png")
- **Location**: `plots/` directory (automatically created)
- **Behavior**: Only saves plots during training (`--train` flag), not during classification

```bash
# Examples with plot saving
python sweet_pepper_segementation_solution1.py --train --save_plots                    # Basic training with plots
python sweet_pepper_classification_solution1.py --train --C 5.0 --save_plots          # Custom parameters with plots
python sweet_pepper_classification_solution2.py --train --margin 1.5 --save_plots      # GMM training with plots
```

## Features and Algorithms

### Feature Extraction
- **LAB Color Space**: Perceptually uniform color representation
- **HSV Histograms**: Color distribution features
- **LBP (Local Binary Patterns)**: Texture features
- **Feature Scaling**: StandardScaler for GMM models
- **PCA**: Dimensionality reduction for gradient boosting

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision and recall
- **Precision-Recall Curves**: Visual performance analysis
- **Classification Reports**: Detailed per-class metrics

### Model Types
- **Parametric**: Custom LogReg, SVM
- **Non-parametric**: KNN
- **Probabilistic**: GMM
- **Ensemble**: Gradient Boosting, Random Forest

## Utility Modules

### `libs/data_utils.py`
- Data loading functions for different datasets
- Color space conversion (RGB → LAB, RGB → HSV)
- Image preprocessing and normalization
- Timing decorators for performance monitoring

### `libs/logreg_utils.py`
- Custom logistic regression implementation
- Precision-recall curve plotting (display and save functions)
- Segmentation mask visualization
- Performance timing utilities

### `libs/svm_utils.py`
- HSV histogram extraction
- LBP feature computation
- Feature combination utilities
- GMM prediction functions
- Image processing pipelines

### `libs/model_utils.py`
- Model saving and loading utilities
- Comprehensive metrics calculation with optional plot saving
- Progress tracking for long operations
- Evaluation utilities for different model types

## Dependencies

- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **scikit-image**: Image processing and feature extraction
- **matplotlib**: Visualization and plotting
- **tqdm**: Progress bars for long operations

this readme.md was made with modern technology :p