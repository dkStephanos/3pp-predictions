import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def add_net_avg_shooting_pct(data_path='./data/shooting_stats.csv', output_path='./data/shooting_stats_extended.csv'):
    """
    Adds a column for net average shooting percentage for October-November data and saves the extended DataFrame.

    This function reads the NBA shooting statistics, calculates the net average shooting percentage
    based on the October-November data, adds this new statistic as a column to the DataFrame, 
    and then saves the extended DataFrame to a new CSV file.

    Parameters:
    data_path (str): The path to the CSV file containing the original data. 
                     Defaults to './data/shooting_stats.csv'.
    output_path (str): The path where the extended CSV file will be saved.
                       Defaults to './data/shooting_stats_extended.csv'.

    Returns:
    None: The function saves the extended DataFrame to a CSV file and does not return anything.
    """

    # Read the data from the CSV file
    df = pd.read_csv(data_path)

    # Calculating the net average shooting percentage for October-November
    shot_types = ['lwr_paint', 'upr_paint', 'mid', 'three_non_cnr', 'three_cnr', 'ft']
    df['net_avg_pct_oct_nov'] = sum(df[f'{shot}_pct_oct_nov'] * df[f'{shot}_shots_oct_nov'] for shot in shot_types) / df[[f'{shot}_shots_oct_nov' for shot in shot_types]].sum(axis=1)

    # Save the extended DataFrame to a new CSV file
    df.to_csv(output_path, index=False)

    return df


