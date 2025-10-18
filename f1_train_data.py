import pandas as pd

# Window Size, to be used in the functions that calculate the rolling historical averages
WINDOW_SIZE = 5 # MAY NEED CHANGE

# Rolling Averages for avg lap time, std lap time, grid position and final position
def basic_rolling_averages(raw_data):
    # Get the minimum average lap time per race
    raw_data['perRaceMinAvgLapTime'] = raw_data.groupby(['Year', 'raceID'])['avgLapTime_s'].transform('min')
    # Calculate the percentage difference for each entry, 
    raw_data['avgLapTime_s_norm'] = (
        (raw_data['avgLapTime_s'] - raw_data['perRaceMinAvgLapTime']) / raw_data['perRaceMinAvgLapTime']
        )
    # Calculate the normalized standard deviation
    raw_data['stdLapTime_s_norm'] = (raw_data['stdLapTime_s'] / raw_data['perRaceMinAvgLapTime'])

    # Columns we want to calculate the historical mean for:
    target_cols = ['avgLapTime_s_norm', 'stdLapTime_s_norm', 'GridPosition', 'Position']

    for col in target_cols:

        # Group the data by driver:
        driver_stats = raw_data.groupby('Driver')[col]

        # Calculate the expanding mean up to the previous race:
        raw_data[f'Prev_Avg_{col}'] = driver_stats.transform(
        # Calculate the expanding mean. Shift one position to the left so that we do not include the current race
        # to the predictors table (X) and avoid data leakage. We use Pandas' .shift(1) for that.
            lambda x: x.expanding(min_periods = 1).mean().shift(1)
        )

        # Calculate the rolling mean (for a given window of races) up to the previous race:
        raw_data[f'Rolling_Prev_Avg_{col}'] = driver_stats.transform(
        # Calculate the rolling mean. Shift one position to the left so that we do not include the current race
        # to the predictors table (X) and avoid data leakage. We use Pandas' .shift(1) for that.
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
        )
    

# Additional rolling averages, grouped by race
def perRace_rolling_averages(raw_data):
    raw_data['Prev_Avg_Finish_Track'] = raw_data.groupby(['Driver', 'CircuitName'])['Position'].transform(
        lambda x: x.expanding(min_periods = 1).mean().shift(1)
    )

    raw_data['Rolling_Prev_Avg_Finish_Track'] = raw_data.groupby(['Driver', 'CircuitName'])['Position'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
    )


# Additional rolling averages, grouped by team
def perTeam_rolling_averages(raw_data):
    # Find out how the team as a whole is doing in recent races when it comes to average lap times
    raw_data['Rolling_Prev_Avg_TeamPace'] = raw_data.groupby('Team')['avgLapTime_s_norm'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
    )

    # Create a "per_team" average position
    raw_data['perRace_Team_Avg_Pos'] = raw_data.groupby(['Year', 'raceID', 'Team'])['Position'].transform('mean')

    raw_data['Rolling_Prev_Avg_TeamFinalPos'] = raw_data.groupby('Team')['perRace_Team_Avg_Pos'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
    )

# Rolling DNFs
def perDriver_rolling_dnf(raw_data):
    raw_data['Rolling_Prev_DNF_Status'] = raw_data.groupby('Driver')['isDNF'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).sum().shift(1)
    )
    # Fill NA with zeros, aka not DNF
    raw_data['Rolling_Prev_DNF_Status'] = raw_data['Rolling_Prev_DNF_Status'].fillna(0)

    # EXTRA
    # Penalize bad results
    raw_data['BadResult'] = (raw_data['Position'] > 10).astype(int)


# Helper function to gather all historical data
# This function will in-place actions to our dataframe
def collect_historical_data(raw_data):
    # Call all rolling functions
    basic_rolling_averages(raw_data)
    perRace_rolling_averages(raw_data)
    perTeam_rolling_averages(raw_data)
    perDriver_rolling_dnf(raw_data)


def drop_na(data_df): # IMPORTAND, collect_historical_data() must run BEFORE this one
    data_df.dropna(
        subset = ['Prev_Avg_avgLapTime_s_norm', 'Prev_Avg_stdLapTime_s_norm', 'Prev_Avg_GridPosition',
                  'Prev_Avg_Position', 'Rolling_Prev_Avg_avgLapTime_s_norm', 'Rolling_Prev_Avg_stdLapTime_s_norm',
                  'Rolling_Prev_Avg_GridPosition', 'Rolling_Prev_Avg_Position', 'Prev_Avg_Finish_Track',
                  'Rolling_Prev_Avg_Finish_Track', 'Rolling_Prev_Avg_TeamPace', 'Rolling_Prev_Avg_TeamFinalPos',
                  ],
        inplace = True
        )

