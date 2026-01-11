from libs.data_utils import data_loader_q1, normalize_rgb_to_lab
from libs.logreg_utils import MyLogReg
import numpy as np

def main():
    # load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()
    X_train_lab = normalize_rgb_to_lab(X_train)
    X_valid_lab = normalize_rgb_to_lab(X_valid)

    #define parameter grid for manual search
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    num_iterations_list = [50, 100, 200, 500]

    best_score = -np.inf
    best_params = None
    best_model = None

    for lr in learning_rates:
        for n_iter in num_iterations_list:
            model = MyLogReg(learning_rate=lr, num_iterations=n_iter)
            model.fit(X_train_lab, y_train)
            y_pred = model.predict(X_valid_lab)
            accuracy = np.mean(y_pred == y_valid)
            print(f"learning_rate={lr}, num_iterations={n_iter}, val_accuracy={accuracy:.4f}")
            if accuracy > best_score:
                best_score = accuracy
                best_params = {'learning_rate': lr, 'num_iterations': n_iter}
                best_model = model

    print("Best parameters:", best_params)
    print("Best validation accuracy:", best_score)

if __name__ == "__main__":
    main()

#after comparing the masks, with a learning rate of 0.0001 and max iterations of 100, the least amount of yellow pepper is segmented.

'''
learning_rate=1e-05, num_iterations=50, val_accuracy=0.9904
learning_rate=1e-05, num_iterations=100, val_accuracy=0.9905
learning_rate=1e-05, num_iterations=200, val_accuracy=0.9907
learning_rate=1e-05, num_iterations=500, val_accuracy=0.9908
learning_rate=0.0001, num_iterations=50, val_accuracy=0.9908
learning_rate=0.0001, num_iterations=100, val_accuracy=0.9910
learning_rate=0.0001, num_iterations=200, val_accuracy=0.9909
learning_rate=0.0001, num_iterations=500, val_accuracy=0.9908
learning_rate=0.001, num_iterations=50, val_accuracy=0.9908
learning_rate=0.001, num_iterations=100, val_accuracy=0.9906
learning_rate=0.001, num_iterations=200, val_accuracy=0.9901
learning_rate=0.001, num_iterations=500, val_accuracy=0.9890
learning_rate=0.01, num_iterations=50, val_accuracy=0.9889
learning_rate=0.01, num_iterations=100, val_accuracy=0.9877
learning_rate=0.01, num_iterations=200, val_accuracy=0.9870
learning_rate=0.01, num_iterations=500, val_accuracy=0.9863
learning_rate=0.1, num_iterations=50, val_accuracy=0.9818
learning_rate=0.1, num_iterations=100, val_accuracy=0.9848
learning_rate=0.1, num_iterations=200, val_accuracy=0.9785
learning_rate=0.1, num_iterations=500, val_accuracy=0.9867
Best parameters: {'learning_rate': 0.0001, 'num_iterations': 100}
Best validation accuracy: 0.9909636363636364
'''