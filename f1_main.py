import pandas as pd
import fastf1
from f1_downloader import get_season
from f1_train_data import collect_historical_data, drop_na
from f1_future_data import get_next_race, pred_cols
from f1_predictor import predict_winner, class_report

# Current Year. Only works for 2025 for now
CURRENT_YEAR = 2025

# Create a cache repository so that we dont download the same data again and again
# (could be a few hundred MBs)
my_path = r'C:\Users\apost\miniconda3\envs\fastF1_cache'
fastf1.Cache.enable_cache(my_path)

# Download season data
# WARNING: get_season() is a custom function. It downloads specific columns, and it aggregates some
stats_2022 = get_season(2022)
stats_2023 = get_season(2023)
stats_2024 = get_season(2024)
stats_2025 = get_season(2025)

# Use Pandas concat to make a big DataFrame with all stats:
all_stats = pd.concat([stats_2022, stats_2023, stats_2024, stats_2025], ignore_index = True)
# Sort the dataframe chronologically:
all_stats = all_stats.sort_values(by = ['Year', 'raceID']).reset_index(drop = True)
# We will need the 'Winner' -binary- column to create our y_train column
all_stats['Winner'] = (all_stats['Position'] == 1).astype(int) 

# Transform our dataframe to get the relevant data for training (Rolling averages, DNFs, historical positions etc)
collect_historical_data(all_stats) # Operations are in-place, assigning the result to a value would lead to None

# Extract the last race stats, to create the future race stats (the ones we want to predict)
next_race = get_next_race(all_stats)

# Test our future data
# print(next_race)

# Add the future race to our dataframe
full_df = pd.concat([all_stats, next_race], ignore_index = True)
# Repass it from collect_historical_data() so that the stats for the next race (curently set to Null)
# will get populated using historical averages
collect_historical_data(full_df)
# Remove any Null values
drop_na(full_df)

# Define the columns that are gonna be used for training
X_cols = pred_cols()
X_future = full_df[full_df['isPredictionData'] == 1][X_cols]
X_train = full_df[full_df['isPredictionData'] != 1][X_cols]
y_train = full_df[full_df['isPredictionData'] != 1]['Winner']

# For identification purposes
ID_cols = ['Driver', 'Year', 'raceID']
ids = full_df[full_df['isPredictionData'] == 1][ID_cols]
train_ids = full_df[full_df['isPredictionData'] != 1][ID_cols]

# Predict the winner of the next Gran Prix
results = predict_winner(X_train, y_train, X_future, ids)
print(results)

# Get a classification report for our model
print("\n\n")
report = class_report(X_train, y_train, train_ids)
print(report)