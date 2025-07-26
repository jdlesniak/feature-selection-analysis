import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    fig, ax = plt.subplots(figsize=(8, 5))
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
    Plot and return top N feature importances from a fitted RandomForestClassifier.

    Args:
        model (RandomForestClassifier): A trained Random Forest model.
        feature_names (list): List of feature names corresponding to model inputs.
        top_n (int): Number of top features to display.

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance'], sorted by importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(top_n), sorted_importances[::-1], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Random Forest Feature Importances")
    fig.tight_layout()
    plt.show()

    # Return importance dataframe
    return pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances
    })

def plot_xgb_feature_importance(model, feature_names, top_n=20):
    """
    Plot and return top N feature importances from a fitted XGBClassifier.

    Args:
        model (XGBClassifier): A trained XGBoost model.
        feature_names (list): List of feature names corresponding to model inputs.
        top_n (int): Number of top features to display.

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'importance'], sorted by importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(top_n), sorted_importances[::-1], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(sorted_features[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} XGBoost Feature Importances")
    fig.tight_layout()
    plt.show()

    # Return importance dataframe
    return pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances
    })


def plot_feature_ranking_comparison(lasso_df, rf_df, xgb_df, top_n=20, selected_lines=False):
    """
    Compare feature rankings across Lasso, Random Forest, and XGBoost.

    Args:
        lasso_df (pd.DataFrame): Output from plot_lasso_paths with ['feature', 'coef']
        rf_df (pd.DataFrame): Output from plot_rf_feature_importance with ['feature', 'importance']
        xgb_df (pd.DataFrame): Output from plot_xgb_feature_importance with ['feature', 'importance']
        top_n (int): Number of top features (by average rank) to include in plot.
        selected_lines (bool): If True, only highlight pre-selected features.

    Returns:
        pd.DataFrame: Combined rankings dataframe (for further use or export).
    """
    # Selected features to highlight (if selected_lines=True)
    selected = {
        'ghazard_c', 'nbumps5', 'ghazard_b', 'nbumps7', 'nbumps89',
        'seismoacoustic_c', 'nbumps6', 'energy', 'maxenergy', 'seismoacoustic_b'
    }

    # Rank features per model
    lasso_df['lasso_rank'] = lasso_df['coef'].abs().rank(ascending=False)
    rf_df['rf_rank'] = rf_df['importance'].abs().rank(ascending=False)
    xgb_df['xgb_rank'] = xgb_df['importance'].abs().rank(ascending=False)

    # Merge ranks and calculate average
    merged = lasso_df[['feature', 'lasso_rank']].merge(
        rf_df[['feature', 'rf_rank']], on='feature', how='outer'
    ).merge(
        xgb_df[['feature', 'xgb_rank']], on='feature', how='outer'
    )
    merged['avg_rank'] = merged[['lasso_rank', 'rf_rank', 'xgb_rank']].mean(axis=1)
    top_features = merged.nsmallest(top_n, 'avg_rank')

    # Reshape for plotting
    plot_df = top_features.melt(
        id_vars='feature',
        value_vars=['lasso_rank', 'rf_rank', 'xgb_rank'],
        var_name='model',
        value_name='rank'
    )
    plot_df['model'] = plot_df['model'].str.replace('_rank', '').str.upper()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Track lines to include in legend
    legend_handles = []

    for feature in plot_df['feature'].unique():
        subset = plot_df[plot_df['feature'] == feature]
        if selected_lines and feature not in selected:
            line, = ax.plot(subset['model'], subset['rank'], color='lightgray', linestyle='--', linewidth=1)
        else:
            line, = ax.plot(subset['model'], subset['rank'], marker='o', label=feature)
            legend_handles.append(line)

    ax.set_ylabel("Feature Rank (lower is more important)")
    ax.set_title(f"Feature Ranking Comparison Across Models (Top {top_n})")
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Always include legend for visible lines
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

    return None
