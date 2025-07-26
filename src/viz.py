import matplotlib.pyplot as plt
import numpy as np

def plot_lasso_paths(result_dict):
    """
    Plot the Lasso coefficient solution paths from a fitted LogisticRegressionCV model.

    Args:
        result_dict (dict): Dictionary returned from fit_lasso_cv(), containing:
            - 'coefs': array of shape (1, n_features, n_Cs) with coefficient paths
            - 'Cs': array of C values used
            - 'feature_names': list of feature names

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object for further use or saving.
    """
    coefs = result_dict['coefs']
    Cs = result_dict['Cs']
    feature_names = result_dict['feature_names']

    # Reshape coefs to (n_features, n_Cs)
    coefs = coefs[0]  # for class 1

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, feature_name in enumerate(result_dict['feature_names']):
        ax.plot(log_Cs, coefs[i], label=feature_name)

    ax.set_xlabel("log10(C)")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Lasso Solution Paths")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True)
    fig.tight_layout()

    return fig

def plot_rf_feature_importance(model, feature_names, top_n=20):
    """
    Plot the top N feature importances from a fitted RandomForestClassifier.

    Args:
        model (RandomForestClassifier): A trained Random Forest model.
        feature_names (list): List of feature names corresponding to model inputs.
        top_n (int): Number of top features to display.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object for the plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), sorted_importances[::-1], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Random Forest Feature Importances")
    fig.tight_layout()

    return fig

def plot_xgb_feature_importance(model, feature_names, top_n=20):
    """
    Plot the top N feature importances from a fitted XGBClassifier.

    Args:
        model (XGBClassifier): A trained XGBoost model.
        feature_names (list): List of feature names corresponding to model inputs.
        top_n (int): Number of top features to display.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object for the plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), sorted_importances[::-1], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} XGBoost Feature Importances")
    fig.tight_layout()

    return fig
