# core/reporter.py
import os
import csv
import pandas as pd

class Reporter:
    """Logs task-level metrics and prints a neat table row to console."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fields = [
            'round','vehicle_id','data_size','connected_bs','distance_to_bs',
            'server_id','priority',
            'signal_dbm','trans_delay_ms','prop_onboard_ms','prop_wireless_ms','prop_wired_ms',
            'proc_delay_ms','handover_delay_ms','tx_cost','proc_cost',
            'cpu_util','mem_util','storage_util', 'reward'
        ]
        self.csv = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.csv, fieldnames=self.fields)
        self.writer.writeheader()
        self.reward_log = []
        self.epsilon_log = []


    def log_and_print(self, round_id, task, server, delays, costs, utils,
                      distance_to_bs, bs_idx, bs_sig, reward=None):
        row = {
            'round':              round_id,
            'vehicle_id':         task['vehicle_id'],
            'data_size':          round(task['data_size'], 2),
            'connected_bs':       bs_idx,
            'distance_to_bs':     round(distance_to_bs, 2),
            'server_id':          server.id if server else 'DROP',
            'priority':           task['priority'],
            'signal_dbm':         round(delays['sig'], 2),
            'trans_delay_ms':     round(delays['trans'], 2),
            'prop_onboard_ms':    round(delays['prop_onboard'], 2),
            'prop_wireless_ms':   round(delays['prop_wireless'], 2),
            'prop_wired_ms':      round(delays['prop_wired'], 2),
            'proc_delay_ms':      round(delays['proc'], 2),
            'handover_delay_ms':  round(delays['ho'], 2),
            'tx_cost':            round(costs['tx'], 2),
            'proc_cost':          round(costs['proc'], 4),
            'cpu_util':           round(utils['cpu'], 2),
            'mem_util':           round(utils['mem'], 2),
            'storage_util':       round(utils['sto'], 2),
            'reward':             round(reward, 2) if reward is not None else None
        }

        self.writer.writerow(row)
        self.csv.flush()

        # Tabular console output
        print(f"| {round_id:3d} | VE{row['vehicle_id']:3d} "
              f"| BS{bs_idx if bs_idx is not None else -1:2d} "
              f"| {row['server_id']:6s} "
              f"| TX {row['trans_delay_ms']:5.1f}ms "
              f"| PR {row['proc_delay_ms']:5.1f}ms "
              f"| CPU {row['cpu_util']:5.1f}% |"
              f"| Memory {row['mem_util']:5.1f}% |"
              f"| Storage {row['storage_util']:5.1f}% |")

    def close(self):
        self.csv.close()


def export_scheduler_logs(scheduler):
    # Make sure the folder exists
    os.makedirs("output", exist_ok=True)

    # 1. Export reward log
    if hasattr(scheduler, 'reward_log') and scheduler.reward_log:
        reward_df = pd.DataFrame(scheduler.reward_log)
        reward_df.to_csv("output/rewards.csv", index=False)

    # 2. Export epsilon decay log
    if hasattr(scheduler, 'epsilon_log') and scheduler.epsilon_log:
        epsilon_df = pd.DataFrame(scheduler.epsilon_log)
        epsilon_df.to_csv("output/epsilon_decay.csv", index=False)

    # 3. Export action distribution log
    if hasattr(scheduler, 'action_log') and scheduler.action_log:
        action_df = pd.DataFrame(scheduler.action_log)
        action_df.to_csv("output/action_distribution.csv", index=False)
