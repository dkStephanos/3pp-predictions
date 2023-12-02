import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_shooting_stats(data_path='./data/shooting_stats.csv'):
    """
    Normalizes the shooting statistics data from a CSV file.

    This function reads a CSV file containing NBA shooting statistics,
    applies standardization to the numerical columns (excluding player names),
    and returns the normalized DataFrame.

    Parameters:
    data_path (str): The path to the CSV file containing the data. 
                     Defaults to './data/shooting_stats.csv'.

    Returns:
    pd.DataFrame: A DataFrame with the normalized shooting statistics.
    """

    # Read the data from the CSV file
    df = pd.read_csv(data_path)

    # Selecting only the numerical columns (excluding player names)
    numerical_cols = df.columns.drop('Name')

    # Applying StandardScaler to the numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
