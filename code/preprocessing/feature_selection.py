"""
File: FeatureSelection.py
Author: Koi Stephanos
Date: 2023-12-02
Description: 
    This file contains methods for performing feature selection in a regression context.
    It is designed to identify the most significant features for predicting target variables 
    in regression models, particularly for analyzing NBA shooting statistics.

Additional Notes:
    - Depends on scikit-learn for feature selection mechanisms.

Modifications:
    - [Date]: [Description of modifications, if any]

Copyright:
    © 2023 Koi Stephanos. All rights reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from code.preprocessing.normalize import normalize_shooting_stats
from code.constants import TARGET_COL
import pandas as pd

def select_features(n_features_to_select=10):
    """
    Selects the top n features for regression based on Recursive Feature Elimination (RFE).

    Args:
        n_features_to_select (int): The number of features to select.

    Returns:
        tuple: A tuple containing a list of top n selected feature names, 
               and a DataFrame with their rankings and importances.
    """
    df = normalize_shooting_stats()
    X = df.drop(columns=["Name", TARGET_COL])
    y = df[TARGET_COL]

    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    fit = rfe.fit(X, y)

    # Get selected features
    selected_features = X.columns[fit.support_]
    selected_importances = fit.estimator_.coef_

    # Create a DataFrame for selected features
    feature_info = pd.DataFrame({
        'Feature': selected_features,
        'Importance': selected_importances
    })

    return selected_features.tolist(), feature_info
