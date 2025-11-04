# workload/generator.py

import random
import config as cfg

class WorkloadGenerator:
    """
    Generates a cyclic workload each round:
      - vehicle count cycles from min_veh → max_veh in steps of veh_inc
      - data size cycles from min_kb → max_kb in steps of data_inc
      - CPU and memory demands scale per KB via cpu_per_kb & mem_per_kb
      - priorities are sampled per the configured distribution
    """

    def __init__(self):
        # grab the entire workload dict
        self.profile = cfg.WORKLOAD

        # vehicle‐count parameters
        self.min_veh   = self.profile['num_vehicles_range'][0]
        self.max_veh   = self.profile['num_vehicles_range'][1]
        self.veh_inc   = self.profile.get('vehicle_increment', 0)

        # data‐size parameters (KB)
        self.min_kb    = self.profile['data_per_device_kb_range'][0]
        self.max_kb    = self.profile['data_per_device_kb_range'][1]
        self.data_inc  = self.profile.get('data_increment', 0)

        # per‐KB resource scaling
        self.cpu_per_kb= self.profile.get('cpu_per_kb', 0)
        self.mem_per_kb= self.profile.get('mem_per_kb', 0)

        # priority sampling setup
        prio_dist      = self.profile['priority_distribution']
        self.prio_keys    = list(prio_dist.keys())
        self.prio_weights = list(prio_dist.values())

    def generate_round(self, round_idx):
        # compute span lengths
        veh_span   = (self.max_veh - self.min_veh) + self.veh_inc
        data_span  = (self.max_kb  - self.min_kb)  + self.data_inc

        # wrap‐around step values
        veh_step   = (self.veh_inc   * round_idx) % veh_span
        data_step  = (self.data_inc  * round_idx) % data_span

        # cycle count & size
        count      = self.min_veh + veh_step
        size       = self.min_kb  + data_step

        # build tasks
        tasks = []
        for vid in range(count):
            prio    = random.choices(
                         population=self.prio_keys,
                         weights=self.prio_weights,
                         k=1
                     )[0]

            cpu_req = int(size * self.cpu_per_kb)
            mem_req = size * self.mem_per_kb

            task = {
                'vehicle_id':     vid,
                'round':          round_idx,
                'data_size':      size,            # KB
                'cpu_demand':     cpu_req,         # MIPS
                'memory_demand':  mem_req,         # MB
                'storage_demand': self.profile['storage_demand'],
                'priority':       prio
            }
            tasks.append(task)

        return tasks
