
import os, csv, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from math import ceil
import argparse

# Show total I/O duration per rank and operation
def plot_time_per_operation(df, filepath, job_id):

    colors = {'write': 'blue', 'open': 'green', 'close': 'red', 'read': 'orange'} 
    
    fig, axes = plt.subplots(1, 4, figsize=(8, 3.5)) 

    for ax, operation in zip(axes, ['write','read', 'open', 'close']):
        df_subset = df[df['op'] == operation].groupby('rank')['dur'].sum().reset_index()
        df_subset.plot(kind='barh', y='dur', x='rank', color=colors[operation], ax=ax, width=0.8)
        ax.set_xlabel('Total I/O duration (sec)')
        ax.set_ylabel('Rank')
        ax.set_yticks(range(0, df_subset['rank'].max(), ceil(df_subset['rank'].max()/10)))
        ax.set_title(operation.capitalize()) 
        ax.get_legend().remove()
        ax.set_xlim(0, df_subset['dur'].max())

    plt.tight_layout()
    plt.show()
    plt.close()
    plt.clf()
    return
 
# Show bandwidth per rank and operation 
def plot_bandwidth_per_rank(df, filepath, job_id):

    df = df[df['op'].isin(['write', 'read'])]
    df['bw'] = np.where(df['dur'] != 0, df['len'] / (df['dur'] * 2**20), 0)
    colors = {'write': 'blue', 'read': 'orange'}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
     
    for ax, operation in zip(axes, ['write','read']):
        df_subset = df[df['op'] == operation].groupby('rank')['bw'].mean().reset_index()
        df_subset.plot(kind='barh', y='bw', x='rank', color=colors[operation], ax=ax, width=0.8)

        ax.set_xlabel('Total Bandwidth (MiB/second)')
        ax.set_ylabel('Rank')
        ax.set_yticks(range(0, df_subset['rank'].max(), ceil(df_subset['rank'].max()/10)))
        ax.set_title(operation.capitalize()) 
        ax.get_legend().remove()

    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

    return

# Plot each I/O event per rank during time
def plot_temporal(df, filepath, job_id):

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') 
    df['dur'] = pd.to_timedelta(df['dur'], unit='s')
    df['start'] = df['timestamp'] - df['dur']
    df['start'] = pd.to_datetime(df['start'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Denver')
    df['start'] = df['start'].dt.tz_localize(None)

    colors = {'write': 'blue', 'open': 'green', 'close': 'red', 'read': 'orange'} 
    df['color'] = df['op'].map(colors)

    fig, ax = plt.subplots(figsize=(18, 3.5))

    ax.barh(y=df['rank'], width=df['dur'], left=df['start'], color=df['color'])
    ax.set_xlabel('Time of the day')
    ax.set_ylabel('Rank')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'[:-3]))
    ax.set_yticks(range(0, df['rank'].max(), ceil(df['rank'].max()/10)))
    legend_labels = [plt.Line2D([0], [0], color=color, linewidth=3, linestyle='-') for op, color in colors.items()]
    ax.legend(legend_labels, colors.keys(), loc='upper right')
    first_date = df['start'].iloc[0]
    ax.set_title(first_date.strftime("Date: " + '%Y-%m-%d'))
    ax.set_ylim(df['rank'].min()-0.5, df['rank'].max()+0.5)
    min_time = df['start'].min()
    max_time = df['start'].max()

    plt.show()
    plt.clf()
    plt.close()

    return

def plot_temporal_points(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') 
    df['dur'] = pd.to_timedelta(df['dur'], unit='s')
    df['start'] = df['timestamp'] - df['dur']
    df['start'] = pd.to_datetime(df['start'], unit='s').dt.tz_localize('UTC').dt.tz_convert('America/Denver')
    df['start'] = df['start'].dt.tz_localize(None)

    colors = {'write': 'blue', 'open': 'green', 'close': 'red', 'read': 'orange'} 
    df['color'] = df['op'].map(colors)
    fig, ax = plt.subplots(figsize=(18, 5))

    # Plot bars for reads and writes
    for op in ['read', 'write']:
        op_data = df[df['op'] == op]
        ax.barh(y=op_data['rank'], width=op_data['dur'], left=op_data['start'], color=op_data['color'])

    # Plot points for opens and closes 
    op_data = df[df['op'] == 'open']
    ax.scatter(op_data['start'], op_data['rank'], color=op_data['color'], label=op, zorder=5, marker="2")
    op_data = df[df['op'] == 'close']
    ax.scatter(op_data['start'], op_data['rank'], color=op_data['color'], label=op, zorder=5, marker="1") 

    ax.set_xlabel('Time of the day')
    ax.set_ylabel('Rank')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'[:-3]))
    ax.set_yticks(range(0, df['rank'].max() + 1, ceil(df['rank'].max()/10)))

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=op) for op, color in colors.items() if op in ['read', 'write']]
    legend_elements += [plt.Line2D([0], [0], marker='2', color=colors['open'], markersize=10, markerfacecolor=colors['open'], label='open')]
    legend_elements += [plt.Line2D([0], [0], marker='1', color=colors['close'], markersize=10, markerfacecolor=colors['close'], label='close')]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_ylim(df['rank'].min()-0.5, df['rank'].max()+0.5)
    first_date = df['start'].iloc[0]
    ax.set_title(first_date.strftime("Date: %Y-%m-%d"))

    plt.show()
    plt.clf()
    plt.close()

    return

# Plot the accumulated time without processing I/O operations 
def io_straggler(df):

    df_read_write = df[df['op'].isin(['read', 'write'])]
    df = df[df['op'].isin(['open', 'close'])]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['start'] = df['timestamp'] - pd.to_timedelta(df['dur'], unit='s')

    intervals = []
    current_id = 1
    for rank in df['rank'].unique():
        current_open = None
        current_id = 1
        for index, row in df[df['rank'] == rank].iterrows():
            if row['op'] == 'open':
                if current_open is None:
                    current_open = row['start']
                else:
                    intervals.append((row['rank'], current_id, (row['start'] - current_open)))
                    current_id += 1
                    current_open = row['start']
            else:
                if current_open is not None:
                    intervals.append((row['rank'], current_id, row['start'] - current_open))
                    current_open = None
                    current_id = 1

    computations = df_read_write.groupby('rank')['dur'].sum().reset_index()
    result = pd.DataFrame(intervals, columns=['rank', 'id', 'interval'])
    result['interval'] = result['interval'].dt.total_seconds()

    df_plot = pd.merge(result.groupby('rank')['interval'].sum().reset_index(), computations, on='rank')
    df_plot['communication'] = df_plot['interval'] - df_plot['dur']
    
    # Get top-3
    top_ranks = df_plot.sort_values('communication', ascending=False)

    # Plot results
    fig, ax = plt.subplots(figsize=(6, 5))
    # ax.barh(df_plot['rank'], df_plot['communication'])
    for index, row in df_plot.iterrows():
        color = 'red' if row['rank'] in top_ranks['rank'].head(3).values else 'blue'
        ax.barh(row['rank'], row['communication'], color=color)
        
    ax.set_yticks(range(0, df['rank'].max() + 1, ceil(df['rank'].max()/10)))
    ax.set_xlabel('Accumulated time between operations (sec)')
    ax.set_ylabel('Rank')
    ax.set_ylim(df['rank'].min()-0.5, df['rank'].max()+0.5)

    plt.show()
    plt.clf()
    plt.close()

    return

# Read file and generate visualizations
def generate_visualizations(df, filepath): 
    # Get basic info about each Job:
    local_df = pd.DataFrame()
    for i in df.job_id.unique():
        
        local_df = df[df['job_id'] == i].copy()
        plot_time_per_operation(local_df, filepath, i)
        plot_bandwidth_per_rank(local_df, filepath, i)
        plot_temporal_points(local_df.copy())
        # io_straggler(local_df)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='Input Darshan-LDMS file in CSV.', type=str, default="", required=True)
    parser.add_argument('-outpath', help='Filepath for the output summary', type=str, default="", required=True)
    args = parser.parse_args()

    df_all = pd.read_csv(args.input, engine="pyarrow")
    start_time_exec = time.time()
    generate_visualizations(df_all, args.outpath)   
    end_time_exec = time.time()
    print("Execution time:", end_time_exec - start_time_exec, "seconds")