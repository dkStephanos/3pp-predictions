import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_summary_table(data_path='./data/shooting_stats_extended.csv'):
    """
    Generates a summary statistics table for the extended NBA shooting statistics.

    Parameters:
    data_path (str): The path to the CSV file containing the extended data. 
                     Defaults to './data/shooting_stats_extended.csv'.

    Returns:
    pd.DataFrame: A DataFrame with the summary statistics for each column, titled 'Summary Statistics'.
    """

    # Read the extended data from the CSV file
    df = pd.read_csv(data_path)

    # Calculate summary statistics
    summary_stats = df.describe().rename_axis('Summary Statistics')

    return summary_stats


def plot_shooting_summary(data_path='./data/shooting_stats_extended.csv'):
    """
    Plots a graph with randomly sampled labels using the extended NBA shooting statistics,
    and adds an average percentage axis for True Shooting Percentage (TS%).

    Parameters:
    data_path (str): The path to the CSV file containing the extended data. 
                     Defaults to './data/shooting_stats_extended.csv'.
    """

    # Read the extended data from the CSV file
    df = pd.read_csv(data_path)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='total_attempts', y='ts_pct', data=df)

    # Determining the indices for top and bottom 1%
    top_1_pct_indices = df['ts_pct'].nlargest(int(len(df) * 0.01)).index
    bottom_1_pct_indices = df['ts_pct'].nsmallest(int(len(df) * 0.01)).index

    # Randomly sampling indices for six more 1% segments, excluding top and bottom 1%
    random_indices = np.random.choice(df.index.difference(top_1_pct_indices.union(bottom_1_pct_indices)), 
                                      size=int(len(df) * 0.06), replace=False)

    # Combining indices for labeling
    label_indices = np.concatenate([top_1_pct_indices, bottom_1_pct_indices, random_indices])

    # Adding labels to the selected samples
    for index in label_indices:
        row = df.loc[index]
        plt.text(row['total_attempts'], row['ts_pct'], row['Name'], 
                 horizontalalignment='left', size='small', color='black', weight='semibold')

    # Plotting average TS% axis
    avg_ts_pct = df['ts_pct'].mean()
    plt.axhline(avg_ts_pct, color='blue', linestyle='--')
    plt.text(df['total_attempts'].max(), avg_ts_pct, f'Avg TS%: {avg_ts_pct:.2f}', 
             verticalalignment='bottom', horizontalalignment='right', color='blue', weight='bold')

    plt.title('Player TS% vs Total Shot Volume (Random Samples Labeled)')
    plt.xlabel('Total Shot Attempts')
    plt.ylabel('True Shooting Percentage (TS%)')
    plt.grid(True)

    plt.show()

