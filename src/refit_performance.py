from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def refit_logistic_from_lasso(lasso_result, X_train, y_train):
    """
    Refit an unpenalized logistic regression using only the non-zero features from Lasso.

    Args:
        lasso_result (dict): Output from fit_lasso_cv().
        X_train (pd.DataFrame): Training features.
        y_train (array-like): Binary training labels.

    Returns:
        model (LogisticRegression): Fitted sklearn logistic regression model.
        summary_df (pd.DataFrame): Statsmodels summary with coef, std err, z, and p-value.
    """
    # Identify selected features from Lasso
    coefs = np.array(lasso_result['coef_best'])
    feature_names = np.array(lasso_result['feature_names'])
    selected_mask = coefs != 0
    selected_features = feature_names[selected_mask]

    if len(selected_features) == 0:
        raise ValueError("Lasso selected no features.")

    # Filter training data to selected features
    X_selected = X_train[selected_features]

    # Refit logistic regression with no penalty (maximum likelihood)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=50000)
    model.fit(X_selected, y_train)

    return model


def plot_model_roc_auc_comparison(logit_model, rf_model, xgb_model, X_test, y_test):
    """
    Plot ROC curves and AUC scores for logistic, random forest, and xgboost models.

    Args:
        logit_model (LogisticRegression): Fitted logistic model using Lasso-selected features.
        rf_model (RandomForestClassifier): Fitted Random Forest model.
        xgb_model (XGBClassifier): Fitted XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (array-like): True labels for test set.
    """

    # Get predictions
    y_prob_logit = logit_model.predict_proba(X_test[logit_model.feature_names_in_])[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Compute ROC curves
    fpr_logit, tpr_logit, _ = roc_curve(y_test, y_prob_logit)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

    # Compute AUCs
    auc_logit = auc(fpr_logit, tpr_logit)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_logit, tpr_logit, label=f"Logistic (AUC = {auc_logit:.3f})", linewidth=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})", linewidth=2)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
