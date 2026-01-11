from libs.pca_rf_utils import MyPCA
from libs.logreg_utils import timing
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def load_q2_data():
    with open('data/Q2_BG_dict.pkl', 'rb') as f:
        bg = pickle.load(f)
    with open('data/Q2_SP_dict.pkl', 'rb') as f:
        sp = pickle.load(f)
    X_train = np.concatenate([np.stack(bg['train']), np.stack(sp['train'])], axis=0)
    y_train = np.concatenate([np.zeros(len(bg['train'])), np.ones(len(sp['train']))])
    X_valid = np.concatenate([np.stack(bg['validation']), np.stack(sp['validation'])], axis=0)
    y_valid = np.concatenate([np.zeros(len(bg['validation'])), np.ones(len(sp['validation']))])
    return X_train, y_train, X_valid, y_valid

@timing
def main():
    X_train, y_train, X_valid, y_valid = load_q2_data()

    # Flatten images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)

    # Grid search parameters
    n_components_list = [10, 20, 40]
    n_estimators_list = [50, 100, 200]
    max_depth_list = [None, 10, 20]

    best_score = -np.inf
    best_params = None

    for n_components in n_components_list:
        pca = MyPCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_valid_pca = pca.transform(X_valid_flat)

        # Set up parameter grid for RandomForest
        param_grid = {
            'n_estimators': n_estimators_list,
            'max_depth': max_depth_list
        }
        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        grid.fit(X_train_pca, y_train)
        y_pred = grid.predict(X_valid_pca)
        acc = accuracy_score(y_valid, y_pred)
        print(f"PCA: {n_components}, RF best params: {grid.best_params_}, val acc: {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_params = {
                'n_components': n_components,
                **grid.best_params_
            }

    print("Best parameters:", best_params)
    print("Best validation accuracy:", best_score)

if __name__ == "__main__":
    main()


'''
It took 10.02 seconds to process your input image
(.venv) PS C:\Users\pinhe\Desktop\Machine Learning Course\Assignment> python pca_rf_gridsearch.py
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.0s
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.0s
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.0s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.1s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.2s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.0s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.2s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.2s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.0s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.4s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.2s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.2s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.5s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.4s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.4s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.4s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.4s
PCA: 10, RF best params: {'max_depth': 10, 'n_estimators': 100}, val acc: 0.9491
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.0s
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.1s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.0s
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.1s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.2s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.2s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.2s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.0s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.2s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.0s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.0s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.2s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.5s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.5s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.2s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.5s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.4s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.4s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.4s
PCA: 20, RF best params: {'max_depth': 10, 'n_estimators': 50}, val acc: 0.9455
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.1s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.1s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.1s
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.1s
[CV] END ....................max_depth=None, n_estimators=50; total time=   0.1s
[CV] END ......................max_depth=10, n_estimators=50; total time=   0.1s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.3s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.3s
[CV] END ...................max_depth=None, n_estimators=100; total time=   0.3s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.3s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.1s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.3s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.1s
[CV] END ......................max_depth=20, n_estimators=50; total time=   0.1s
[CV] END .....................max_depth=10, n_estimators=100; total time=   0.3s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.7s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.7s
[CV] END ...................max_depth=None, n_estimators=200; total time=   0.7s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.6s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.3s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.3s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.6s
[CV] END .....................max_depth=20, n_estimators=100; total time=   0.3s
[CV] END .....................max_depth=10, n_estimators=200; total time=   0.7s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.6s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.6s
[CV] END .....................max_depth=20, n_estimators=200; total time=   0.6s
PCA: 40, RF best params: {'max_depth': None, 'n_estimators': 100}, val acc: 0.9400
Best parameters: {'n_components': 10, 'max_depth': 10, 'n_estimators': 100}
Best validation accuracy: 0.9490909090909091
It took 19.39 seconds to process your input image

'''