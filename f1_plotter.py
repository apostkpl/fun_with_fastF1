import fastf1
import fastf1.plotting as f1plt
import matplotlib.pyplot as plt
import pandas as pd

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