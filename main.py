import os, csv, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from math import ceil
import argparse

warnings.filterwarnings('ignore') 

class Job:

    def __init__(self, job, ranks, nodes, users, filename, exe):
        
        self.job = job
        self.ranks = ranks
        self.nodes = nodes
        self.users = users
        self.filename = filename
        self.exe = exe

def app_phase(df, output_file, self):
    write_to_file("---------------------------------------")
    write_to_file("EXECUTION SUMMARY PER APPLICATION PHASE:")
    write_to_file("---------------------------------------")

# Calculate and write general statistics in a file
def get_statistics(df, output_file, self):

    with open(output_file, 'w') as f:

        def write_to_file(*args):
            print(" ".join(map(str, args)), file=f, flush=True)

        write_to_file("---------------------------------------")
        write_to_file("JOB CHARACTERISTICS:")
        write_to_file("---------------------------------------")
        write_to_file("Job ID:", self.job)
        write_to_file(len(self.ranks), "Rank (s):", sorted(self.ranks))
        write_to_file(len(self.nodes), "Node (s):", sorted(self.nodes))
        write_to_file("User ID:", self.users)
        write_to_file("Directory:", self.exe)
        write_to_file("Modules collected:", df['module'].unique())
        write_to_file("Module events (MOD):", list(df.type).count('MOD'))
        write_to_file("Meta events (MET):", list(df.type).count('MET'))

        df_read = df[df['op'] == "read"]
        df_write = df[df['op'] == "write"]
        df_open = df[df['op'] == "open"]
        df_close = df[df['op'] == "close"]

        write_to_file("---------------------------------------")
        write_to_file("I/O OPERATIONS:")
        write_to_file("---------------------------------------")

        exec_time = round(df['end'].max() - df['start'].min(), 5)
        write_to_file("Total I/O makespan:", exec_time, "seconds")
        write_to_file("Cumulative I/O duration:", round(df['dur'].sum(), 5), "seconds")
        write_to_file("Bandwidth (MiB/second):", round((df['len'].sum() / exec_time) / (1024 ** 2), 5))
        write_to_file("IOPS:", round(len(df)/exec_time, 5), "\n")

        current_op = None
        phase_start = None
        total_durations = {'read': 0, 'write': 0, 'open': 0, 'close': 0}

        def update_total_duration(op, phase_start, phase_end, length):
            if current_op is not None and current_op == op:
                total_durations[op] += (phase_end - phase_start)

        for index, row in df.iterrows():
            if current_op is None or current_op != row['op']:
                update_total_duration(current_op, phase_start, row['end'], row['len'])
                current_op = row['op']
                phase_start = row['start']

        # Get the last phase
        update_total_duration(current_op, phase_start, row['end'], row['len'])

        pivot_df = df.pivot_table(index=None, columns='op', values='len', aggfunc='sum')
        for op, duration in total_durations.items():
            write_to_file(f'Duration {op}s: {round(duration, 4)} seconds')

        write_to_file("\n# of reads: ", len(df_read))
        write_to_file("Cumulative I/O duration:",round(df_read['dur'].sum(),2))
        write_to_file("Total bytes:",round(df_read['len'].sum() / (1024 ** 2)),  "MiB")
        write_to_file("Min data size per rank:", round(df_read.groupby('rank')['len'].agg('sum').min() / (1024 ** 2)), "MiB")
        write_to_file("Max data size per rank:", round(df_read.groupby('rank')['len'].agg('sum').max() / (1024 ** 2)), "MiB")
        # write_to_file("Bandwidth (MiB/second):", round((df_read['len'].sum() / total_durations['read']) / (1024 ** 2), 2))
        # write_to_file("IOPS:", round(len(df_read)/total_durations['read'], 2))

        write_to_file("\n# of writes: ", len(df_write))
        write_to_file("Cumulative I/O duration:",round(df_write['dur'].sum(), 2))
        write_to_file("Total bytes:",round(df_write['len'].sum() / (1024 ** 2)),  "MiB")
        write_to_file("Min data size per rank:", round(df_write.groupby('rank')['len'].agg('sum').min() / (1024 ** 2)), "MiB")
        write_to_file("Max data size per rank:", round(df_write.groupby('rank')['len'].agg('sum').max() / (1024 ** 2)), "MiB")
        # write_to_file("Bandwidth (MiB/second):", round((df_write['len'].sum() / total_durations['write']) / (1024 ** 2),2))
        # write_to_file("Mean IOPS:", round((df_write.groupby('rank').size() / df_write.groupby('rank')['dur'].sum()).mean(), 2))
        # write_to_file("IOPS:", round(len(df_write)/total_durations['write'], 2))

        write_to_file("\n# of opens: ", len(df_open))
        write_to_file("\n# of closes: ", len(df_close))

        # IMBALANCE METRICS:
        write_to_file("---------------------------------------")
        write_to_file("I/O PROGRESS RATE:")
        write_to_file("---------------------------------------")
        # Get difference between execution time and time processing I/O per rank
        
        df_idle = df.groupby('rank')['dur'].sum().reset_index()
        df_idle.columns = ['Rank ID', 'Rank I/O Time (sec)']
        df_idle['Total Bytes'] = df.groupby('rank')['len'].agg('sum')
        df_idle['Rank I/O Time (sec)'] = round(df_idle['Rank I/O Time (sec)'], 2)
        # df_idle['Total I/O Time - Rank I/O Time'] = round(exec_time - df_idle['Rank I/O Time'], 2)
        df_idle = df_idle.sort_values(by='Rank I/O Time (sec)', ascending=False)
        
        num_ranks = len(self.ranks)
        average = df['dur'].sum() / num_ranks
        write_to_file("- Average:", round(average, 2), "seconds")
        std = np.std(df_idle['Rank I/O Time (sec)'])
        write_to_file("- Standard Deviation", round(std, 2), "seconds")
        it = df_idle['Rank I/O Time (sec)'].max() - average
        write_to_file("- Imbalance Time:", round(it, 2), "seconds")
        pi = ((df_idle['Rank I/O Time (sec)'].max() / average) - 1) * 100
        write_to_file("- Percent Imbalance:", round(pi, 2), "%")
        # ti = exec_time - df.groupby('rank')['dur'].sum().max()
        # write_to_file("- Time Interval:", round(ti, 2), "seconds")
        # ip = (it / df_idle['Rank I/O Time (sec)'].max()) * (num_ranks / (num_ranks - 1))
        # write_to_file("- Imbalance Percentage:", round(ip, 2), "%")

        write_to_file("---------------------------------------")
        write_to_file("SUMMARY PER RANK: \n(ordered by higher I/O time - all ops)")
        write_to_file("---------------------------------------")
        df.loc[:, 'start'] = pd.to_datetime(df['start'], unit='s').dt.round('S')
        df.loc[:, 'end'] = pd.to_datetime(df['end'], unit='s').dt.round('S')
        # write_to_file(df_idle)
        write_to_file(df_idle.to_string(index=False))
        
# Read CSV file, define jobs characteristics and calculate load metrics
def main(filename, filepath): 

    df_report = pd.read_csv(filename, engine="pyarrow")

    # Get basic info about each Job:
    local_df = pd.DataFrame()
    for i in df_report.job_id.unique():
        
        local_df = df_report[df_report['job_id'] == i].copy()
        job = Job(i, local_df['rank'].unique(), local_df['ProducerName'].unique(),local_df['uid'].unique(), 
            local_df['file'].unique(), local_df['exe'].unique())

        local_df.loc[:, 'start'] = local_df['timestamp'] - local_df['dur']
        local_df.loc[:, 'end'] = local_df['timestamp']
        
        # Job characteristics and statistics:  
        output_file = filename.replace(".csv", ".txt")
        get_statistics(local_df, output_file, job)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='Input Darshan-LDMS file in CSV.', type=str, default="", required=True)
    parser.add_argument('-outpath', help='Filepath for the output summary', type=str, default="", required=True)
    args = parser.parse_args()

    start_time_exec = time.time()
    main(args.input, args.outpath)   
    end_time_exec = time.time()
    print("Execution time:", end_time_exec - start_time_exec, "seconds")