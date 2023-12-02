def add_net_avg_shooting_pct(df):
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

    # Calculating the net average shooting percentage for October-November
    shot_types = ['lwr_paint', 'upr_paint', 'mid', 'three_non_cnr', 'three_cnr', 'ft']
    df['net_avg_pct_oct_nov'] = sum(df[f'{shot}_pct_oct_nov'] * df[f'{shot}_shots_oct_nov'] for shot in shot_types) / df[[f'{shot}_shots_oct_nov' for shot in shot_types]].sum(axis=1)

    return df

def add_true_shooting_percentage(df):
    """
    Adds the True Shooting Percentage (TS%) to the shooting statistics.

    True Shooting Percentage is a measure of shooting efficiency that takes into account
    field goals, 3-point field goals, and free throws. This function calculates the TS%
    using the formula:

        TS% = Total Points / (2 * (Total Field Goal Attempts + 0.44 * Free Throw Attempts))

    The formula accounts for the different values of free throws, two-pointers, and three-pointers.

    Parameters:
    data_path (str): Path to the CSV file containing the shooting statistics.

    Returns:
    pd.DataFrame: The DataFrame with the added True Shooting Percentage column.
    """

    # Calculating total attempts (sum of all shot attempts across different types)
    attempt_cols = ['lwr_paint_shots_oct_nov', 'upr_paint_shots_oct_nov', 'mid_shots_oct_nov', 
                    'three_non_cnr_shots_oct_nov', 'three_cnr_shots_oct_nov', 'ft_shots_oct_nov']
    df['total_attempts'] = df[attempt_cols].sum(axis=1)

    # Calculating total points
    # Points are calculated based on the shooting percentages and attempts for each type of shot.
    # Free throws (ft) count for 1 point each, three-point shots for 3 points each, and all others for 2 points each.
    df['total_points'] = (df['lwr_paint_pct_oct_nov'] * df['lwr_paint_shots_oct_nov'] * 2 +
                          df['upr_paint_pct_oct_nov'] * df['upr_paint_shots_oct_nov'] * 2 +
                          df['mid_pct_oct_nov'] * df['mid_shots_oct_nov'] * 2 +
                          df['three_non_cnr_pct_oct_nov'] * df['three_non_cnr_shots_oct_nov'] * 3 +
                          df['three_cnr_pct_oct_nov'] * df['three_cnr_shots_oct_nov'] * 3 +
                          df['ft_pct_oct_nov'] * df['ft_shots_oct_nov'])

    # Calculating True Shooting Percentage (TS%)
    # TS% is calculated as the ratio of total points to the weighted shot attempts (including free throws)
    df['ts_pct'] = df['total_points'] / (2 * (df['total_attempts'] + 0.44 * df['ft_shots_oct_nov']))

    return df


