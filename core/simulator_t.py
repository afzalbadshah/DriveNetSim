
import math
import os
import pandas as pd
import config as cfg
from entities.server import Server
from workload.generator import WorkloadGenerator
from mobility.manager import MobilityManager
from core.reporter import Reporter
from core.reporter import export_scheduler_logs
from config.metrics import (
    calculate_transmission_delay,
    calculate_tx_cost,
    calculate_proc_cost,
    wireless_delay,
    wired_delay
)


class Simulator:
    """Simulator orchestrates workload, mobility, scheduling, allocation, and reporting."""
    def __init__(self, scheduler, output_dir="output"):
        self.scheduler = scheduler
        self.servers = []

        max_veh = cfg.WORKLOAD['num_vehicles_range'][1]
        for i in range(max_veh):
            self.servers.append(Server(f"VE_{i}", **cfg.VEHICULAR_EDGE))
        for i in range(cfg.NUM_BASE_STATIONS):
            self.servers.append(Server(f"BS_{i}", **cfg.BS_EDGE))
        self.servers.append(Server("CL_0", **cfg.CLOUD))

        self.scheduler.servers = self.servers
        self.generator = WorkloadGenerator()
        self.mobility = MobilityManager()
        self.reporter = Reporter(f"{output_dir}/results.csv")
       #self.scheduler.save_logs(output_dir=f"{output_dir}/logs")

    def run(self, train_mode: bool = False):
        if train_mode:
            print(f"[TRAINING MODE] Starting training for {cfg.SIMULATION_ROUNDS} rounds...")

        self.mobility.initialize_positions()

        for r in range(cfg.SIMULATION_ROUNDS):
            action_counts = {0: 0, 1: 0, 2: 0}
            if train_mode:
                print(f"[TRAINING MODE] Round {r+1}/{cfg.SIMULATION_ROUNDS}")

            tasks_per_server = {srv.id: 0 for srv in self.servers}
            tasks = self.generator.generate_round(r)

            for task in tasks:
                vid = task['vehicle_id']
                ve_id = f"VE_{vid}"
                server = self.scheduler.select_server(task, self.servers, self.mobility)

                if server.id.startswith('VE_'):
                    action_counts[0] += 1
                elif server.id.startswith('BS_'):
                    action_counts[1] += 1
                else:
                    action_counts[2] += 1

                bw = server.bandwidth_mbps
                if server.id.startswith("CL_"):
                    bs_idx, _ = self.mobility.connected_bs[ve_id]
                    bs_server = next(s for s in self.servers if s.id == f"BS_{bs_idx}")
                    trans1 = calculate_transmission_delay(task['data_size'], bs_server.bandwidth_mbps, bs_server)
                    trans2 = calculate_transmission_delay(task['data_size'], bw, server)
                    trans = trans1 + trans2
                else:
                    trans = calculate_transmission_delay(task['data_size'], bw, server)

                proc = task['cpu_demand'] / server.cpu * 10
                base_reward = -0.01 * (trans + proc)

                if train_mode:
                    self.scheduler.update(base_reward, task, self.servers, self.mobility)
                else:
                    frag, leftover = server.allocate_until_full(
                        task, r,
                        util_threshold=(
                            cfg.VE_UTIL_THRESHOLD if server.id.startswith("VE_")
                            else cfg.BS_UTIL_THRESHOLD if server.id.startswith("BS_")
                            else 100.0
                        )
                    )
                    if frag:
                        tasks_per_server[server.id] += 1

                    bs_idx, bs_sig = self.mobility.connected_bs[ve_id]
                    pos = self.mobility.positions[ve_id]
                    distance = self.mobility.circular_distance(pos, cfg.BS_POSITIONS[bs_idx])

                    delays = {
                        'trans': trans,
                        'prop_onboard': cfg.VE_PROP_DELAY_MS,
                        'prop_wireless': 0.0 if server.id.startswith("VE_") else wireless_delay(distance),
                        'prop_wired': 0.0 if not server.id.startswith("CL_") else wired_delay(cfg.BS_CLOUD_DISTANCE_M),
                        'proc': proc,
                        'ho': getattr(server, 'handover_delay_ms', 0.0),
                        'sig': bs_sig
                    }
                    tier = ('vehicular_edge' if server.id.startswith("VE_")
                            else 'bs_edge' if server.id.startswith("BS_")
                            else 'cloud')
                    data_size = frag['data_size'] if frag else task['data_size']
                    costs = {
                        'tx': calculate_tx_cost(data_size, tier),
                        'proc': calculate_proc_cost(proc, tier)
                    }
                    utils = {
                        'cpu': server.get_cpu_util(),
                        'mem': server.get_mem_util(),
                        'sto': server.get_storage_util()
                    }

                    self.scheduler.update(base_reward, task, self.servers, self.mobility)
                    self.reporter.log_and_print(r, task, server, delays, costs, utils, distance, bs_idx, bs_sig)

            if train_mode:
                total = sum(action_counts.values())
                if not hasattr(self.scheduler, 'action_log'):
                    self.scheduler.action_log = []
                self.scheduler.action_log.append({
                    'episode': r,
                    'edge': action_counts[0] / total if total else 0,
                    'regional': action_counts[1] / total if total else 0,
                    'cloud': action_counts[2] / total if total else 0
                })

            for srv in self.servers:
                srv.release_completed(r)
            self.mobility.update_positions()

        if train_mode:
            print(f"[TRAINING MODE] Training complete. Total Q-states: {len(self.scheduler.Q)}")
            if hasattr(self.scheduler, 'action_log'):
                os.makedirs('output', exist_ok=True)
                pd.DataFrame(self.scheduler.action_log).to_csv('output/action_distribution.csv', index=False)
        else:
            self.reporter.close()
            export_scheduler_logs(self.scheduler)

        self.scheduler.save()
