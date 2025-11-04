import os
import pickle
import random
import numpy as np
import pandas as pd
import config as cfg

class AIScheduler:
    Q_FILENAME = "q_table.pkl"

    def __init__(self):
        # Load or initialize Q-table
        self.Q = self._load_q_table()
        self.servers = None
        self.vehicle_bs = {}
        self.last_action = {}

        # Exploration parameters
        self.epsilon = 1.0           # full exploration initially
        self.min_epsilon = 0.01      # floor for epsilon
        self.decay = 0.997            # decay factor per round
        # Learning parameters
        self.alpha = 0.3             # learning rate
        self.gamma = 0.9             # discount factor

        # Logs for export
        self.reward_log = []         # each entry: {'round': int, 'vehicle_id': int, 'reward': float, ...}
        self.epsilon_log = []        # each entry: {'round': int, 'epsilon': float}

    def _load_q_table(self):
        if os.path.exists(self.Q_FILENAME):
            with open(self.Q_FILENAME, "rb") as f:
                return pickle.load(f)
        return {}
    
    def set_train_mode(self, is_training: bool):
        self.train_mode = is_training
        self.epsilon = 0.5 if is_training else 0.0

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
            # initialize action-values for VE, BS, CL
            self.Q[state] = np.zeros(3)

    def calculate_reward(self, ve, bs, cl, selected, sig, task):
        ve_cpu = ve.get_cpu_util()
        ve_mem = ve.get_mem_util()
        bs_cpu = bs.get_cpu_util()
        bs_mem = bs.get_mem_util()

        print(f"[Debug:] VE CPU: {ve_cpu}, MEM: {ve_mem}, ")
        print(f"[Debug:] BS CPU: {bs_cpu}, MEM: {bs_mem}, Signal: {sig}")

        reward = 0

        # --- Reward or penalize based on selected server ---
        if selected == ve:
            if ve_cpu < 70 and ve_mem < 70:
                reward += 30
            elif ve_cpu < 90 and ve_mem < 90:
                reward += 10
            else:
                reward -= 10

        elif selected == bs:
            if ve_cpu >= 70 or ve_mem >= 70:
                if bs_cpu < 70 and bs_mem < 70 and sig >= cfg.BS_SIG_THRESHOLD:
                    reward += 10
                else:
                    reward -= 20
            else:
                reward -= 20  # BS selected when VE was not overloaded

        elif selected == cl:
            if ve_cpu >= 90 and ve_mem >= 90 and bs_cpu >= 90 and bs_mem >= 90:
                reward += 5
            else:
                reward -= 30

        # --- Penalties for skipping VE or BS when usable ---

        # Skipping VE unnecessarily
        if selected != ve and ve.can_allocate(task) and ve_cpu < 70 and ve_mem < 70:
            reward -= 40

        # Skipping BS unnecessarily
        if (selected == cl and
            bs.can_allocate(task) and
            bs_cpu < 70 and bs_mem < 70 and
            sig >= cfg.BS_SIG_THRESHOLD):
            reward -= 40

        return reward


    def select_server(self, task, servers, mobility):
        self.servers = servers
        vid = task['vehicle_id']
        state = self._get_state(task, mobility)
        self._init_state(state)

        ve = self._get_server(servers, "VE", vehicle_id=vid)
        conn_idx, sig = mobility.connected_bs[f"VE_{vid}"]
        bs = self._get_server(servers, "BS", bs_id=conn_idx)
        cl = self._get_server(servers, "CL")

        feasible = [ve.can_allocate(task),
                    bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD,
                    cl.can_allocate(task)]
        if not any(feasible):
            feasible = [False, False, True]

        # Epsilon-greedy action
        if random.random() < self.epsilon:
            action = random.choice([i for i, ok in enumerate(feasible) if ok])
        else:
            qvals = self.Q[state].copy()
            qvals = [q if feasible[i] else -np.inf for i, q in enumerate(qvals)]
            action = int(np.argmax(qvals))

        selected = [ve, bs, cl][action]
        task['_last_action'] = action

        # Compute reward and perform Q-update
        reward = self.calculate_reward(ve, bs, cl, selected, sig, task)
        self.update(reward, task, servers, mobility)

        # Track vehicle's last action and BS
        self.last_action[vid] = action
        if action == 1:
            self.vehicle_bs[vid] = selected

        return selected

    def update(self, reward, task, servers, mobility):
        vid = task['vehicle_id']
        curr_round = task.get('round', -1)
        prev = task.get('_prev_action', -1)
        curr = task.get('_last_action', -1)

        # Switching penalty
        if prev != -1 and curr != prev:
            reward -= 10

        # Q-learning update
        state = self._get_state(task, mobility)
        self._init_state(state)
        action_idx = curr if curr >= 0 else int(np.argmax(self.Q[state]))
        best_next = np.max(self.Q[state])
        td_target = reward + self.gamma * best_next
        self.Q[state][action_idx] += self.alpha * (td_target - self.Q[state][action_idx])

        # Update prev action
        task['_prev_action'] = curr

        # Decay epsilon once per round after 30
        if curr_round >= 30:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        # Log reward and epsilon
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
        with open(self.Q_FILENAME, "wb") as f:
            pickle.dump(self.Q, f)

    def save_logs(self, output_dir="results/logs"):
        os.makedirs(output_dir, exist_ok=True)
        if self.reward_log:
            pd.DataFrame(self.reward_log).to_csv(f"{output_dir}/rewards.csv", index=False)
        if self.epsilon_log:
            pd.DataFrame(self.epsilon_log).to_csv(f"{output_dir}/epsilon_decay.csv", index=False)
