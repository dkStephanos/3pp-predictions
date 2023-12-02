from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from code.preprocessing.normalize import normalize_shooting_stats
from code.constants import NUMERIC_COLS
from code.utils import project_cols
import pandas as pd

def perform_feature_selection(
    C: float = 0.2
) -> pd.DataFrame:
    """
    Performs feature selection using a Linear Support Vector Classification (LinearSVC) model
    with L1-based feature selection. This function fits a LinearSVC model to the data, 
    selects the most important features, and optionally aligns the result with a clustering result.
    
    Args:
        C (float, optional): The regularization parameter for the LinearSVC model. Defaults to 0.001.
    
    Returns:
        pd.DataFrame: A dataframe with selected features, and optionally with cluster labels.
    """
    
    # Normalize input df
    stats_df = normalize_shooting_stats()

    # Project the data to the set of numeric columns
    X = project_cols(stats_df, NUMERIC_COLS)
    
    # Get the target variable
    y = stats_df["win"]
    
    # Fit a LinearSVC model with L1 regularization
    lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(X, y)
    
    # Select features using the fitted model
    model = SelectFromModel(lsvc, prefit=True)
    X_new = X.loc[:, model.get_support()]
    X_new.loc[:, "win"] = y
    
    # If a clustering result is provided, align the feature selection result with it
    if cluster_df is not None:
        X_new = X_new[X_new.index.isin(cluster_df.index)]
        X_new['cluster'] = cluster_df['cluster']
    elif 'cluster' in list(stats_df.columns):
        # Fall back to original cluster assignement if there
        X_new.loc[:, 'cluster'] = stats_df['cluster']
    
    return X_new