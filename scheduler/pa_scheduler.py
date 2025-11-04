import os
import pickle
import random
import numpy as np
import pandas as pd
import config as cfg

class AIScheduler:
    Q_FILENAME = "q_table.pkl"

    def __init__(self):
        self.train_mode = False
        self.Q = self._load_q_table()
        self.servers = None
        self.vehicle_bs = {}
        self.last_action = {}

        self.epsilon = 0.5
        self.min_epsilon = 0.01
        self.decay = 0.95

        self.alpha = 0.3
        self.gamma = 0.9

        self.reward_log = []
        self.epsilon_log = []

    def set_train_mode(self, is_training: bool):
        self.train_mode = is_training
        self.epsilon = 0.5 if is_training else 0.0

    def _load_q_table(self):
        if os.path.exists(self.Q_FILENAME):
            with open(self.Q_FILENAME, "rb") as f:
                return pickle.load(f)
        return {}

    def _get_server(self, servers, tier, vehicle_id=None, bs_id=None):
        if tier == "VE":
            return next(s for s in servers if s.id == f"VE_{vehicle_id}")
        elif tier == "BS":
            return next(s for s in servers if s.id == f"BS_{bs_id}")
        elif tier == "CL":
            return next(s for s in servers if s.id.startswith("CL_"))
        return None

  
    def _get_state(self, task, mobility):
        vid = task['vehicle_id']
        conn_idx, _ = mobility.connected_bs[f"VE_{vid}"]
        last_bs = self.vehicle_bs.get(vid)
        last_bs_idx = -1 if last_bs is None else int(last_bs.id.split("_")[1])

        ve = self._get_server(self.servers, "VE", vehicle_id=vid)
        bs = self._get_server(self.servers, "BS", bs_id=conn_idx)

        ve_cpu = int(min(ve.get_cpu_util(), 100) // 5)
        ve_mem = int(min(ve.get_mem_util(), 100) // 5)
        bs_cpu = int(min(bs.get_cpu_util(), 100) // 5)
        bs_mem = int(min(bs.get_mem_util(), 100) // 5)

        prev = task.get('_prev_action', -1)
        priority = task.get("priority", 1)
        data_bucket = int(task.get("data_size", 0) // 5000)

        return (last_bs_idx, conn_idx, ve_cpu, ve_mem, bs_cpu, bs_mem, prev, priority, data_bucket)

    def _init_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(3)

    def calculate_reward(self, ve, bs, cl, selected, sig, task):
        priority = task.get("priority", 1)
        signal_bonus = 2 if sig >= cfg.BS_SIG_THRESHOLD else -2

        
        ve_cpu = ve.get_cpu_util()
        ve_mem = ve.get_mem_util()
        bs_cpu = bs.get_cpu_util()
        bs_mem = bs.get_mem_util()

        print(f"[Debug:] VE CPU: {ve_cpu}, MEM: {ve_mem}, ")
        print(f"[Debug:] BS CPU: {bs_cpu}, MEM: {bs_mem}, Signal: {sig}")
        
        # Base reward based on correct tier for priority
        reward = 0

        if priority == 0:
            if selected == ve and ve.can_allocate(task):
                reward = 20
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                reward = 10 + signal_bonus
            elif selected == cl:
                reward = -10
            else:
                reward = -20

        elif priority == 1:
            if selected == ve and ve.can_allocate(task):
                reward = 10
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                reward = 6 + signal_bonus
            elif selected == cl:
                reward = -2
            else:
                reward = -6

        elif priority == 2:
            if selected == ve and ve.can_allocate(task):
                reward = 10  # Lower priority still rewarded if VE is chosen and feasible
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                reward = 8 + signal_bonus
            elif selected == cl:
                reward = 5
            else:
                reward = -5

        # 💥 Penalize skipping VE when it's underutilized and available
        if selected != ve and ve.can_allocate(task) and ve_cpu < 70 and ve_mem < 70:
            reward -= 40

        # 💥 Penalize skipping BS when it’s underutilized and signal is OK
        if (selected == cl and
            bs.can_allocate(task) and
            bs_cpu < 70 and bs_mem < 70 and
            sig >= cfg.BS_SIG_THRESHOLD):
            reward -= 40

        return reward      


    def select_server(self, task, servers, mobility):
        #print(f"[DEBUG] select_server: train_mode={self.train_mode}, epsilon={self.epsilon}")
        #print(f"[DEBUG] train_mode={self.train_mode}, epsilon={self.epsilon}")
        #print(f"[DEBUG] Q-values = {self.Q.get(self._get_state(task, mobility), [])}")

        self.servers = servers
        vid = task['vehicle_id']
        state = self._get_state(task, mobility)
        self._init_state(state)

        ve = self._get_server(servers, "VE", vehicle_id=vid)
        conn_idx, sig = mobility.connected_bs[f"VE_{vid}"]
        bs = self._get_server(servers, "BS", bs_id=conn_idx)
        cl = self._get_server(servers, "CL")

        if bs.get_cpu_util() > 90 or bs.get_mem_util() > 90:
            prev_bs = self.vehicle_bs.get(vid)
            if prev_bs and prev_bs.can_allocate(task):
                bs = prev_bs

    
        feasible = [ve.can_allocate(task),
                    bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD,
                    cl.can_allocate(task)]
        if not any(feasible):
            feasible = [False, False, True]

        if self.train_mode and random.random() < self.epsilon:
            action = random.choice([i for i, ok in enumerate(feasible) if ok])
        else:
            qvals = self.Q[state].copy()
            qvals = [q if feasible[i] else -np.inf for i, q in enumerate(qvals)]
            action = int(np.argmax(qvals))

        selected = [ve, bs, cl][action]
        task['_last_action'] = action

        if self.train_mode:
            reward = self.calculate_reward(ve, bs, cl, selected, sig, task)
            self.update(reward, task, mobility)

        self.last_action[vid] = action
        if selected == bs:
            self.vehicle_bs[vid] = bs

        return selected

    def update(self, reward, task, mobility):
        if not self.train_mode:
            return

        vid = task['vehicle_id']
        curr_round = task.get('round', -1)
        prev = task.get('_prev_action', -1)
        curr = task.get('_last_action', -1)

        if prev != -1 and curr != prev:
            reward -= 10

        state = self._get_state(task, mobility)
        self._init_state(state)

        action_idx = curr if curr >= 0 else int(np.argmax(self.Q[state]))
        best_next = np.max(self.Q[state])
        td_target = reward + self.gamma * best_next
        self.Q[state][action_idx] += self.alpha * (td_target - self.Q[state][action_idx])

        task['_prev_action'] = curr

        if curr_round >= 30:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        self.reward_log.append({
            'round': curr_round,
            'vehicle_id': vid,
            'reward': reward,
            'action': curr,
            'prev_action': prev
        })
        self.epsilon_log.append({
            'round': curr_round,
            'epsilon': self.epsilon
        })

    def save(self):
        # Always save Q-table
        #os.makedirs("results/logs", exist_ok=True)
        with open( self.Q_FILENAME, "wb") as f:
            pickle.dump(self.Q, f)

    def save_logs(self, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)

        if self.train_mode:
            if self.reward_log:
                pd.DataFrame(self.reward_log).to_csv(f"{output_dir}/rewards.csv", index=False)
            if self.epsilon_log:
                pd.DataFrame(self.epsilon_log).to_csv(f"{output_dir}/epsilon_decay.csv", index=False)

        # Always save Q-table again
        self.save()
