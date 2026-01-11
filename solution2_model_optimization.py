from libs.data_utils import data_loader_q1, normalize_rgb_to_lab
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from libs.logreg_utils import timing

@timing
def main():
    #load data
    X_train, y_train, X_eval, y_eval, X_valid, y_valid = data_loader_q1()

    #convert all data to lab color space
    X_train_lab = normalize_rgb_to_lab(X_train)

    #define parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    #create SVC instance
    svc = SVC(probability=True)

    #create grid search object
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

    #fit grid search
    grid_search.fit(X_train_lab, y_train)

    print("best parameters:", grid_search.best_params_)
    print("best cross-validation score:", grid_search.best_score_)

if __name__ == "__main__":
    main()


'''
python solution2_model_optimization.py
Fitting 3 folds for each of 24 candidates, totalling 72 fits
[CV] END ..................C=0.1, gamma=scale, kernel=linear; total time=  55.7s
[CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time=  57.5s
[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=  57.7s
[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=  57.9s
[CV] END ...................C=0.1, gamma=0.01, kernel=linear; total time=  57.9s
[CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time= 1.1min
[CV] END ..................C=0.1, gamma=scale, kernel=linear; total time= 1.2min
[CV] END ..................C=0.1, gamma=scale, kernel=linear; total time= 1.3min
[CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time= 1.4min
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time= 1.4min
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time= 1.7min
[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=  53.9s
[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=  55.0s
[CV] END ....................C=0.1, gamma=0.1, kernel=linear; total time=  55.1s
[CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time= 2.1min
[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=  59.9s
[CV] END .......................C=1, gamma=scale, kernel=rbf; total time=  44.8s
[CV] END ......................C=0.1, gamma=1, kernel=linear; total time=  55.5s
[CV] END ......................C=0.1, gamma=1, kernel=linear; total time= 1.3min
[CV] END .......................C=1, gamma=scale, kernel=rbf; total time=  58.0s
[CV] END .......................C=1, gamma=scale, kernel=rbf; total time=  55.1s
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time= 1.2min
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time= 1.2min
[CV] END ........................C=1, gamma=0.01, kernel=rbf; total time= 1.3min
[CV] END ....................C=1, gamma=scale, kernel=linear; total time= 3.0min
[CV] END ....................C=1, gamma=scale, kernel=linear; total time= 2.9min
[CV] END ....................C=1, gamma=scale, kernel=linear; total time= 2.8min
[CV] END .....................C=1, gamma=0.01, kernel=linear; total time= 2.9min
[CV] END .....................C=1, gamma=0.01, kernel=linear; total time= 2.8min
[CV] END .....................C=1, gamma=0.01, kernel=linear; total time= 2.8min
[CV] END ......................C=1, gamma=0.1, kernel=linear; total time= 2.7min
[CV] END ......................C=1, gamma=0.1, kernel=linear; total time= 2.7min
[CV] END ......................C=1, gamma=0.1, kernel=linear; total time= 3.0min
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=16.7min
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=17.1min
[CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=17.2min
[CV] END ........................C=1, gamma=1, kernel=linear; total time= 2.8min
[CV] END ........................C=1, gamma=1, kernel=linear; total time= 2.8min
[CV] END ........................C=1, gamma=1, kernel=linear; total time= 2.7min
[CV] END ......................C=10, gamma=scale, kernel=rbf; total time=  43.0s
[CV] END ......................C=10, gamma=scale, kernel=rbf; total time=  41.7s
[CV] END ......................C=10, gamma=scale, kernel=rbf; total time=  42.9s
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=22.0min
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=22.0min
[CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=22.8min
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time= 1.7min
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time= 1.6min
[CV] END .......................C=10, gamma=0.01, kernel=rbf; total time= 1.8min
[CV] END ...................C=10, gamma=scale, kernel=linear; total time=14.8min
[CV] END ...................C=10, gamma=scale, kernel=linear; total time=15.5min
[CV] END ...................C=10, gamma=scale, kernel=linear; total time=15.7min
[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=15.2min
[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=15.7min
[CV] END ....................C=10, gamma=0.01, kernel=linear; total time=16.7min
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=47.6min
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=51.4min
[CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=51.7min
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=20.4min
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=19.5min
[CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=20.6min
[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=15.4min
[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=16.0min
[CV] END .....................C=10, gamma=0.1, kernel=linear; total time=16.7min
[CV] END .......................C=10, gamma=1, kernel=linear; total time=14.0min
[CV] END .......................C=10, gamma=1, kernel=linear; total time=14.3min
[CV] END .......................C=10, gamma=1, kernel=linear; total time=13.9min
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=82.9min
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=83.0min
[CV] END ...........................C=1, gamma=1, kernel=rbf; total time=82.8min
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=66.7min
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=64.6min
[CV] END ..........................C=10, gamma=1, kernel=rbf; total time=67.3min
best parameters: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
best cross-validation score: 0.9927909084285544
'''