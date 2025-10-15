import fastf1
import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone, timedelta

# Upload delay. Fastf1 typically takes 1-2 hours before uploading a current race's data. I use a 48 hours delay to make it safer
# but feel free to change it to a lower value if necessary/useful
UPLOAD_DELAY = timedelta(hours = 48) # MAY NEED CHANGE

# Window Size, to be used in the functions that calculate the rolling historical averages
WINDOW_SIZE = 5 # MAY NEED CHANGE

# Helper function to load a race
def get_race(year, race_number):
    try:
        schedule = fastf1.get_event_schedule(year)
        if race_number < 0 or race_number > len(schedule):
            raise ValueError(f'Error. Race number is out of bounds. Season {year} had {len(schedule)} races')
        session = fastf1.get_session(year, race_number, 'R')
        return session
    except Exception as e:
        raise Exception(f'An error occurred while getting the {year} session: {e}') from e

# Helper function to get a whole season
def get_season(year):
    try:
        # Check if the year given is indeed in the fastf1 database
        if year < 2018 or year > 2025:
            raise ValueError(f'Error, Season {year} not available in this database. Available seasons are: 2018-2025')
        schedule = fastf1.get_event_schedule(year)
        # Remove future races from the schedule, using an upload delay to be safe
        safe_cutoff_date = pd.to_datetime(datetime.now(timezone.utc) - UPLOAD_DELAY)
        if 'Session5DateUtc' in schedule.columns:
            schedule_dates = schedule['Session5DateUtc']
            # Check if the dates are indeed timezone aware (may differ by season):
            if schedule_dates.dt.tz == None:
                # If the dates are NOT datetime aware
                schedule_dates = schedule_dates.dt.tz_localize('UTC')
            else:
                # If the dates are datetime aware
                schedule_dates = schedule_dates.dt.tz_convert('UTC')
            schedule = schedule[schedule_dates < safe_cutoff_date].copy()
        # Create a new dataframe for the current season
        avg_season_stats = pd.DataFrame()
        # Repeat the same procedure for every other race in the current session
        total_races = schedule['RoundNumber'].dropna().astype(int).tolist()
        for i in total_races:
            # The first index/row is the test race of the season
            if i == 0:
                continue
            
            current = get_race(year, i)
            current.load(laps = True, telemetry = False, weather = False, messages = False)
            # Get the name of the current race
            circuit_name = current.event['EventName']
            # Aggregate important stats into a dataframe
            avg_current = current.laps.groupby('Driver').agg(
                avgLapTime = ('LapTime', 'mean'),
                stdLapTime = ('LapTime', 'std'),
                lapsCompleted = ('LapNumber', 'max'),
                Team = ('Team', 'first') # Although not "aggregating" it, I need the team name from the .laps() property for later
                ).reset_index()

            # Add the circuit name to my aggregated "avg_current" DataFrame
            avg_current['CircuitName'] = circuit_name
            # Check if a driver had a DNF in the current race
            dnf_types = ['Retired', 'Accident', 'Mechanical', 'Engine', 'Brakes', 'Damage', 'Collision', 'Hydraulics']
            results_df = current.results[['Abbreviation', 'GridPosition', 'Position', 'Status']].rename(
                columns = {'Abbreviation': 'Driver'}
            )
            results_df['isDNF'] = results_df['Status'].apply(lambda stat: 1 if stat in dnf_types else 0)
            results_df.drop(columns = ['Status'], inplace = True)
            # Convert Timedelta to seconds
            avg_current['avgLapTime_s'] = avg_current['avgLapTime'].dt.total_seconds()
            avg_current['stdLapTime_s'] = avg_current['stdLapTime'].dt.total_seconds()
            avg_current = avg_current.drop(['avgLapTime', 'stdLapTime'], axis = 1)

            # Merge the aggregated df with the results DNF status
            avg_current = avg_current.merge(
                results_df, # includes 'Abbreviation', 'GridPosition', 'Position', plus the new 'isDNF' columns ('Status' col dropped)
                how = 'left',
                on = 'Driver'
            )

            # Keep track of the race number and the current year
            avg_current['raceID'] = i
            avg_current['Year'] = year

            # Add the new rows to the stats dataframe
            avg_season_stats = pd.concat([avg_season_stats, avg_current], ignore_index = True)

        # Return the whole season stats dataframe
        return avg_season_stats
        
    except Exception as e:
        raise Exception(f'An error occurred while getting the {year} season: {e}') from e


# Helper function that plots the lap times for a driver
def plot_lap_times(driver_name, laps_df, ax, color = 'steelblue', alpha = 1):
    ax.plot(laps_df.pick_drivers(driver_name)['LapNumber'], laps_df.pick_drivers(driver_name)['LapTime'], color = color,
           alpha = alpha, lw = 2, label = driver_name)
    ax.set_title(laps_df.attrs['name'] + '\n', fontweight = 'bold', fontsize = 16)
    ax.set_xlabel('Lap Number', fontweight = 'bold')
    ax.set_ylabel('Lap Time', fontweight = 'bold')
    ax.set_xticks(range(0, len(laps_df.pick_drivers(driver_name)['LapNumber']), 1))
    ax.grid()
    ax.legend(title = 'Driver Name:\n',
              title_fontproperties={'weight': 'bold'},
              prop={'weight': 'bold'}, shadow = True, fancybox = True
             )

# Helper function that makes a barplot of the final results for a certain race
def get_final_results(results_df, race_df, ax):
    # Get the positions and names, sort them by position
    final = results_df.loc[:, ['Position', 'BroadcastName']].sort_values(by = 'Position').reset_index(drop = True)
    #final.index = range(1, len(results_df['BroadcastName']) + 1)
    # Get the team colors using the get_team_color API call
    team_colors = [f1plt.get_team_color(team, race_df) for team in results_df['TeamName']]

    # Plot the results
    ax.bar(final['BroadcastName'], final['Position'], color = team_colors, ec = 'black')
    ax.set_title(results_df.attrs['name'] + "\n", fontweight = 'bold', fontsize = 16)
    ax.tick_params(axis = 'x',labelrotation = 75)
    ax.set_ylabel('Position', fontweight = 'bold')
    ax.set_yticks(range(1, len(results_df['Position']) + 1))
    ax.grid(axis = 'y', alpha = 0.3)

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

    # Drop potential new null values:
    raw_data.dropna(
        subset = ['Prev_Avg_avgLapTime_s_norm', 'Prev_Avg_stdLapTime_s_norm', 'Prev_Avg_GridPosition',
                  'Prev_Avg_Position', 'Rolling_Prev_Avg_avgLapTime_s_norm', 'Rolling_Prev_Avg_stdLapTime_s_norm',
                  'Rolling_Prev_Avg_GridPosition', 'Rolling_Prev_Avg_Position'],
                  inplace = True
                  )

# Additional rolling averages, grouped by race
def perRace_rolling_averages(raw_data):
    raw_data['Prev_Avg_Finish_Track'] = raw_data.groupby(['Driver', 'CircuitName'])['Position'].transform(
        lambda x: x.expanding(min_periods = 1).mean().shift(1)
    )

    raw_data['Rolling_Prev_Avg_Finish_Track'] = raw_data.groupby(['Driver', 'CircuitName'])['Position'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
    )

    raw_data.dropna(subset = ['Prev_Avg_Finish_Track', 'Rolling_Prev_Avg_Finish_Track'],inplace = True)


# Additional rolling averages, grouped by team
def perTeam_rolling_averages(raw_data):
    # Find out how the team as a whole is doing in recent races when it comes to average lap times
    raw_data['Rolling_Prev_Avg_TeamPace'] = raw_data.groupby('Team')['avgLapTime_s_norm'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
    )

    # Create a "per_team" df for summed "Position"s
    per_team_stats = raw_data.groupby(['Year', 'raceID', 'Team'])['Position'].mean().reset_index()
    # Merge the new summed Positions (will rename them) with the original df
    per_team_stats.rename(columns = {'Position': 'perRace_Team_Avg_Pos'}, inplace = True)
    raw_data = raw_data.merge(
        per_team_stats[['Year', 'raceID', 'Team', 'perRace_Team_Avg_Pos']],
        on=['Year', 'raceID', 'Team'],
        how='left'
        )

    raw_data['Rolling_Prev_Avg_TeamFinalPos'] = raw_data.groupby('Team')['perRace_Team_Avg_Pos'].transform(
        lambda x: x.rolling(window = WINDOW_SIZE, min_periods = 1).mean().shift(1)
    )

    raw_data.dropna(
        subset = ['Rolling_Prev_Avg_TeamPace', 'Rolling_Prev_Avg_TeamFinalPos'],
        inplace = True
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
