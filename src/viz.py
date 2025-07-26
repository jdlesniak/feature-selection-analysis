import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd

def plot_lasso_paths(result_dict):
    """
    Plot coefficient paths for L1-penalized logistic regression, with optimal C marker.
    - Non-zero features use solid lines; zeroed features use dashed lines.
    - All features appear in the legend with correct color/style.
    - Returns a DataFrame of coefficients at best_C, sorted by abs(coef).

    Args:
        result_dict (dict): Output from fit_lasso_cv

    Returns:
        pd.DataFrame: DataFrame with ['feature', 'coef'], sorted by abs(coef)
    """
    # Prep base plot
    fig, ax = plt.subplots(figsize=(10, 6))
    log_Cs = -np.log10(result_dict['Cs'])
    best_C_index = np.argmin(np.abs(result_dict['Cs'] - result_dict['best_C']))

    # Build dataframe of feature-level info
    coef_df = pd.DataFrame({
        'feature': result_dict['feature_names'],
        'coef': result_dict['coef_best']
    })
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False).reset_index(drop=True)

    # Assign color and line style
    cmap = colormaps.get_cmap('tab20')
    coef_df['color'] = [cmap(i % 20) for i in range(len(coef_df))]
    coef_df['linestyle'] = coef_df['coef'].apply(lambda c: '-' if c != 0 else '--')

    # Plot all paths with proper style
    for i, row in coef_df.iterrows():
        coef_path = result_dict['coefs'][result_dict['feature_names'].index(row['feature'])]
        ax.plot(log_Cs, coef_path,
                label=row['feature'],
                color=row['color'],
                linestyle=row['linestyle'],
                linewidth=2 if row['coef'] != 0 else 1)

    # Optimal C marker
    ax.axvline(x=-np.log10(result_dict['best_C']), color='gray', linestyle='--', label='Optimal C')

    # Finalize plot
    ax.set_xlabel("-log10(C)")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Lasso Path")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

    return coef_df.drop(['abs_coef','color', 'linestyle'], axis = 1)

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
