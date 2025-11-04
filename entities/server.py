import math
import config as cfg

class Server:
    def __init__(self, id, cpu, memory_mb, storage_gb,
                 bandwidth_mbps, propagation_delay_ms,
                 handover_delay_ms=0.0):
        self.id = id
        # capacities
        self.cpu = cpu                    # MIPS
        self.memory_mb = memory_mb        # MB
        self.storage_gb = storage_gb      # GB
        self.bandwidth_mbps = bandwidth_mbps  # Mbps
        self.propagation_delay_ms = propagation_delay_ms
        self.handover_delay_ms = handover_delay_ms

        # available resources start equal to total capacities
        self.available_cpu = cpu
        self.available_memory = memory_mb
        self.available_storage = storage_gb

        # queue of pending tasks: list of (finish_round, task)
        self._pending = []

    def can_allocate(self, task):
        return (
            self.available_cpu >= task['cpu_demand'] and
            self.available_memory >= task['memory_demand']
        )

    def allocate(self, task, current_round, proc_time_rounds):
        # deduct resources
        self.available_cpu -= task['cpu_demand']
        self.available_memory -= task['memory_demand']
        storage_needed = task.get('storage_demand', 0)
        self.available_storage -= storage_needed

        # ensure at least 1 round occupancy
        finish_round = current_round + max(1, math.ceil(proc_time_rounds))
        self._pending.append((finish_round, task))

    def release_completed(self, current_round):
        survivors = []
        for finish, task in self._pending:
            if finish == current_round:
                # return resources
                self.available_cpu += task['cpu_demand']
                #self.available_memory += task['memory_demand']
                mem = task.get('memory_demand', 0)
                self.available_memory += mem
                storage_needed = task.get('storage_demand', 0)
                self.available_storage += storage_needed
            else:
                survivors.append((finish, task))
        self._pending = survivors

    def get_cpu_util(self):
        used = self.cpu - self.available_cpu
        return (used / self.cpu) * 100 if self.cpu else 0.0

    def get_mem_util(self):
        used = self.memory_mb - self.available_memory
        return (used / self.memory_mb) * 100 if self.memory_mb else 0.0

    def get_storage_util(self):
        used = self.storage_gb - self.available_storage
        return (used / self.storage_gb) * 100 if self.storage_gb else 0.0

    def allocate_until_full(self, task, current_round, proc_factor=10, util_threshold=100.0):
        # 1) raw capacity fraction
        cpu_frac = (self.available_cpu / task['cpu_demand']) if task['cpu_demand'] > 0 else 1.0
        mem_frac = (self.available_memory / task['memory_demand']) if task['memory_demand'] > 0 else 1.0
        stor_frac = 1.0
        if task.get('storage_demand', 0) > 0:
            stor_frac = (self.available_storage / task['storage_demand'])
        raw_frac = min(1.0, cpu_frac, mem_frac, stor_frac)
        if raw_frac <= 0:
            return None, task.copy()

        # 2) util-based fraction cap for CPU
        curr_cpu_util = (self.cpu - self.available_cpu) / self.cpu * 100
        max_cpu_extra = max(0.0, util_threshold - curr_cpu_util)
        per_full_cpu_pct = (task['cpu_demand'] / self.cpu) * 100
        util_frac_cpu = min(1.0, max_cpu_extra / per_full_cpu_pct) if per_full_cpu_pct > 0 else 1.0

        # 3) util-based fraction cap for Memory
        curr_mem_util = (self.memory_mb - self.available_memory) / self.memory_mb * 100
        max_mem_extra = max(0.0, util_threshold - curr_mem_util)
        per_full_mem_pct = (task['memory_demand'] / self.memory_mb) * 100
        util_frac_mem = min(1.0, max_mem_extra / per_full_mem_pct) if per_full_mem_pct > 0 else 1.0

        # combine CPU and memory util caps
        serve_frac_util = min(util_frac_cpu, util_frac_mem)
        serve_frac = min(raw_frac, serve_frac_util)
        if serve_frac <= 0:
            return None, task.copy()

        # 4) build subtask and allocate
        sub = task.copy()
        for k in ('data_size','cpu_demand','memory_demand','storage_demand'):
            sub[k] = task[k] * serve_frac

        self.available_cpu    -= sub['cpu_demand']
        self.available_memory -= sub['memory_demand']
        self.available_storage= max(0, self.available_storage - sub.get('storage_demand', 0))

        proc_time    = sub['cpu_demand'] / self.cpu * proc_factor
        finish_round = current_round + max(1, math.ceil(proc_time))
        self._pending.append((finish_round, sub))

        # leftover
        leftover = None
        if serve_frac < 1.0:
            leftover = task.copy()
            for k in ('data_size','cpu_demand','memory_demand','storage_demand'):
                leftover[k] = task[k] * (1 - serve_frac)

        return sub, leftover
