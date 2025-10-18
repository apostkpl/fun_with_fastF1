import pandas as pd
import numpy as np

CURRENT_GRID = ['VER', 'NOR', 'PIA', 'HUL', 'RUS', 'ALO', 'SAI', 'HAM', 'ALB', 'LEC',
                'ANT', 'HAD', 'GAS', 'STR', 'LAW', 'BEA', 'COL', 'TSU', 'OCO', 'BOR']

def get_next_race(data_df, grid = False):
    next_race = data_df.iloc[-20:, :].copy()
    if next_race['raceID'].iloc[0] < 24: # Hardcoded length for 2025, will change it for a getter function later
        numeric_columns = ['lapsCompleted', 'avgLapTime_s', 'stdLapTime_s', 'GridPosition', 'Position',
                   'perRaceMinAvgLapTime', 'avgLapTime_s_norm', 'stdLapTime_s_norm','Prev_Avg_avgLapTime_s_norm',
                   'Rolling_Prev_Avg_avgLapTime_s_norm', 'Prev_Avg_stdLapTime_s_norm', 'Rolling_Prev_Avg_stdLapTime_s_norm',
                   'Prev_Avg_GridPosition', 'Rolling_Prev_Avg_GridPosition', 'Prev_Avg_Position', 'Rolling_Prev_Avg_Position',
                   'Prev_Avg_Finish_Track', 'Rolling_Prev_Avg_Finish_Track', 'Rolling_Prev_Avg_TeamPace',
                   'perRace_Team_Avg_Pos', 'Rolling_Prev_Avg_TeamFinalPos', 'Rolling_Prev_DNF_Status', 'isDNF', 'BadResult']

        next_race[numeric_columns] = np.nan
        next_race['raceID'] += 1
        next_race['isPredictionData'] = 1
        if grid:
            grid_map = {driver_name: i + 1 for i, driver_name in enumerate(CURRENT_GRID)}
            next_race['GridPosition'] = next_race['Driver'].map(grid_map)
            return next_race
    
    return next_race

def pred_cols(grid = False):
    if grid:
        cols = ['Prev_Avg_GridPosition', 'Prev_Avg_Position', 'Prev_Avg_avgLapTime_s_norm', 'Prev_Avg_stdLapTime_s_norm',
         'Rolling_Prev_Avg_avgLapTime_s_norm', 'Rolling_Prev_Avg_stdLapTime_s_norm',
         'Rolling_Prev_Avg_GridPosition', 'Rolling_Prev_Avg_Position', 'Prev_Avg_Finish_Track',
         'Rolling_Prev_Avg_Finish_Track', 'Rolling_Prev_Avg_TeamPace', 'Rolling_Prev_Avg_TeamFinalPos', 'GridPosition'
         #, 'BadResult'
         ]
        return cols

    cols = ['Prev_Avg_GridPosition', 'Prev_Avg_Position', 'Prev_Avg_avgLapTime_s_norm', 'Prev_Avg_stdLapTime_s_norm',
         'Rolling_Prev_Avg_avgLapTime_s_norm', 'Rolling_Prev_Avg_stdLapTime_s_norm',
         'Rolling_Prev_Avg_GridPosition', 'Rolling_Prev_Avg_Position', 'Prev_Avg_Finish_Track',
         'Rolling_Prev_Avg_Finish_Track', 'Rolling_Prev_Avg_TeamPace', 'Rolling_Prev_Avg_TeamFinalPos'
         #, 'BadResult'
         ]
    return cols

# IMPORTANT :
# MAKE GETTERS FOR raceID and CircuitName