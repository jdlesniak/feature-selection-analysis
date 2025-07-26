from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

### get our util functions and paths
from utils import *
from paths import CLEAN_DATA_DIR, SEED

def fit_lasso_cv(X, y, seed, cv_folds=5, C_grid=None):
    """
    Fit a cross-validated Lasso (L1-penalized) logistic regression and capture solution paths.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Binary target vector (0/1).
        seed (int): Random seed for reproducibility.
        cv_folds (int): Number of cross-validation folds.
        C_grid (list or np.ndarray): Optional grid of inverse regularization strengths (C = 1/lambda).

    Returns:
        dict: {
            'model': fitted LogisticRegressionCV object,
            'Cs': array of C values tested,
            'coefs': array of shape (n_Cs, n_features),
            'best_C': best C value selected,
            'coef_best': coefficients at best C,
            'feature_names': list of feature names
        }
    """
    if C_grid is None:
        C_grid = np.logspace(-4, 4, 100)  # can customize this

    model = LogisticRegressionCV(
        Cs=C_grid,
        cv=cv_folds,
        penalty='l1',
        solver='saga',  
        scoring='accuracy',
        random_state=seed,
        max_iter=50000,
        refit=True
    )

    ## standardize the features in a pipeline
    ## lasso needs standardized features otherwise the scale of
    ## coefficients screws with the estimates
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X, y)

    # Extract model from pipeline
    lasso = pipe.named_steps['logisticregressioncv']

    # Get coefficients along path
    coefs = lasso.coefs_paths_[1]  # class 1 → binary classification
    Cs = lasso.Cs_
    best_C = lasso.C_[0]
    coef_best = lasso.coef_.flatten()

    return {
        'model': lasso,
        'Cs': Cs,
        'coefs': coefs,
        'best_C': best_C,
        'coef_best': coef_best,
        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else [f'x{i}' for i in range(X.shape[1])]
    }

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_rf_cv(X, y, seed, cv_folds=5, n_iter=25):
    """
    Fit a Random Forest classifier with cross-validation and randomized hyperparameter tuning.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Binary target vector.
        seed (int): Random seed for reproducibility.
        cv_folds (int): Number of cross-validation folds.
        n_iter (int): Number of random combinations to try.
        return_search (bool): If True, also return the RandomizedSearchCV object.

    Returns:
        RandomForestClassifier: Best-fit model.
        (optional) RandomizedSearchCV: Full search object with cross-val results.
    """
    n_features = X.shape[1]
    n_samples = X.shape[0]

    # Dynamic search space depending on data dimensions
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.25, 0.5]
    }

    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv_folds,
        random_state=seed,
        verbose=0,
        n_jobs=-1,
        return_train_score=True
    )

    # Fit search
    search.fit(X, y)

    return search.best_estimator_

def fit_xgb_cv(X, y, seed, cv_folds=5, n_iter=25):
    """
    Fit an XGBoost classifier with cross-validation and randomized hyperparameter tuning.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Binary target vector.
        seed (int): Random seed for reproducibility.
        cv_folds (int): Number of cross-validation folds.
        n_iter (int): Number of random parameter combinations to evaluate.
        return_search (bool): If True, also return the RandomizedSearchCV object.

    Returns:
        XGBClassifier: Best-fit model.
        (optional) RandomizedSearchCV: Full search object with CV results.
    """
    param_dist = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.25, 1.0]
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=seed,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv_folds,
        verbose=0,
        n_jobs=-1,
        random_state=seed,
        return_train_score=True
    )

    search.fit(X, y)

    return search.best_estimator_
