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

        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.99

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
        bs_signal = mobility.connected_bs[f"VE_{vid}"][1]

        return (priority, ve_cpu, ve_mem, bs_signal)

    def _init_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(3)

    def _to_pct(self, u):
        return u #* 100.0 if u <= 1.0 else u


    def calculate_reward(self, ve, bs, cl, selected, sig, task, ve_cpu, ve_mem, bs_cpu, bs_mem):
        priority = task.get("priority", 1)
        # half-scale the old ±1 signal bonus to ±0.5
        signal_bonus = 0.5 if sig >= cfg.BS_SIG_THRESHOLD else -0.5
        

        raw = 0.0

        # 1) Hard penalty if VE is overloaded for priorities 2/3 (was -10 → -5)
        if priority in (2, 3) and selected == ve and (ve_cpu >= 50 or ve_mem >= 50):
            raw = -10.0  # VE overload penalty

        # 2) Base rewards by priority
        elif priority == 1:
            if selected == ve and ve.can_allocate(task):
                raw = 10.0 
            elif selected == bs and bs.can_allocate(task) and (ve_cpu > 80 or ve_mem > 80) and sig >= cfg.BS_SIG_THRESHOLD:
                raw = 3.0 + signal_bonus 
            else:
                raw = -10.0  # P1 failed offload penalty, was -8

        elif priority == 2:
            if selected == ve and ve.can_allocate(task) and (ve_cpu < 50 and ve_mem < 50):
                raw = 7  # was 3
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                raw = 3 + signal_bonus
            elif selected in (bs, cl) and (ve_cpu < 50 or ve_mem < 50 ):
                raw = -10  
            else:
                raw = -5  # P2 failed offload penalty, was -5

        elif priority == 3:
            if selected == ve and ve.can_allocate(task) and (ve_cpu < 50 and ve_mem < 50):
                raw = 6  # was 3
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                raw = 4 + signal_bonus  
            elif selected == cl:
                raw = 3  # CL fallback for P3, was 3
            elif selected in (bs, cl) and (ve_cpu < 50 or ve_mem < 50 ):
                raw = -10
            else:
                raw = -2.5  # P3 failed offload penalty, was -5

        # 3) Reduced skip-penalties (were -2 → now -1)
        #if selected != ve and ve.can_allocate(task) and (ve_cpu < 70 and ve_mem < 70):
            #raw -= 3.0  # Should have used VE

        #if selected == cl and bs.can_allocate(task) and (bs_cpu < 70 or bs_mem < 70) and sig >= cfg.BS_SIG_THRESHOLD:
            #raw -= 3.0  # Should have used BS

        # 4) Normalize into roughly [-1..+1]
        return raw / 5.0


    def select_server(self, task, servers, mobility):
        # stash for update()
        self.servers = servers

        vid   = task['vehicle_id']
        state = self._get_state(task, mobility)
        self._init_state(state)

        print(f"[DEBUG] select_server: train_mode={self.train_mode}, epsilon={self.epsilon}")
        print(f"[DEBUG] Q-values = {self.Q.get(state, [])}")

        task['_prev_state']  = state
        task['_prev_action'] = task.get('_last_action', -1)

        # fetch VE / BS / CL
        ve  = self._get_server(servers, "VE", vehicle_id=vid)
        bs_idx, sig = mobility.connected_bs[f"VE_{vid}"]
        bs  = self._get_server(servers, "BS", bs_id=bs_idx)
        cl  = self._get_server(servers, "CL")

        # utilizations (0–100)
        #ve_cpu, ve_mem = map(self._to_pct, (ve.get_cpu_util(), ve.get_mem_util()))
        #bs_cpu, bs_mem = map(self._to_pct, (bs.get_cpu_util(), bs.get_mem_util()))

        ve_cpu = int(min(ve.get_cpu_util(), 100) // 5)
        ve_mem = int(min(ve.get_mem_util(), 100) // 5)
        bs_cpu = int(min(bs.get_cpu_util(), 100) // 5)
        bs_mem = int(min(bs.get_mem_util(), 100) // 5)


        # feasibility mask
        feasible = [
            ve.can_allocate(task),
            bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD,
            cl.can_allocate(task)
        ]
        if not any(feasible):
            feasible = [False, False, True]

        # ε-greedy (only in training)
        if self.train_mode and random.random() < self.epsilon:
            action = random.choice([i for i,ok in enumerate(feasible) if ok])
        else:
            qvals = self.Q[state]
            masked = [q if feasible[i] else -np.inf for i,q in enumerate(qvals)]
            action = int(np.argmax(masked))

        task['_last_action'] = action
        selected = [ve, bs, cl][action]

        print(f"[DEBUG] Selected server: {selected.id} for vehicle {vid}")

        # pre-allocate so next_state sees the bump
        #if self.train_mode and selected.can_allocate(task):
            #selected.allocate_until_full(task, task.get('round',0), util_threshold=1.0)

        # compute reward
        reward = self.calculate_reward(
            ve, bs, cl, selected, sig, task,
            ve_cpu, ve_mem, bs_cpu, bs_mem
        )

        if self.train_mode:
            self.update(reward, task, mobility)

        # keep track for logging / downstream
        self.last_action[vid] = action
        if action == 1:
            self.vehicle_bs[vid] = selected

        return selected

    def update(self, reward, task, mobility):
        # only learn in train mode
        if not self.train_mode:
            return

        # unpack bookkeeping slots
        vid         = task['vehicle_id']
        curr_round  = task.get('round', -1)
        prev_act    = task.get('_prev_action', -1)
        curr_act    = task.get('_last_action', -1)
        prev_state  = task.get('_prev_state')

        print(f"[DEBUG] update(): reward={reward}, prev_state={prev_state}, "
            f"prev_act={prev_act}, curr_act={curr_act}")

        # 1) reduced switching penalty
        if prev_act != -1 and curr_act != prev_act:
            reward -= 2
            print(f"[DEBUG]   switching penalty applied, new reward={reward}")

        # 2) locate the three servers and simulate your allocation step
        ve      = self._get_server(self.servers, "VE", vehicle_id=vid)
        bs_idx, _ = mobility.connected_bs[f"VE_{vid}"]
        bs      = self._get_server(self.servers, "BS", bs_id=bs_idx)
        cl      = self._get_server(self.servers, "CL")
        selected = [ve, bs, cl][curr_act]

        if selected.can_allocate(task):
            selected.allocate_until_full(task, curr_round, util_threshold=1.0)

        # 3) compute the next state (make sure _get_state returns a richer tuple!)
        next_state = self._get_state(task, mobility)
        self._init_state(next_state)
        print(f"[DEBUG]   next_state={next_state}, Q[next_state]={self.Q[next_state]}")

        # 4) Q-learning update, using prev_act
        # 4) Q-learning update, using prev_act vs. curr_act for init
        if prev_act < 0:
            # first time we see this (state,action) pair → set it to its immediate target
            best_next = np.max(self.Q[next_state])
            init_val  = reward + self.gamma * best_next
            self.Q[prev_state][curr_act] = init_val
            print(f"[DEBUG]   first step – initializing Q[{prev_state}][{curr_act}] = {init_val:.4f}")
        else:
            # normal TD update
            best_next = np.max(self.Q[next_state])
            td_target = reward + self.gamma * best_next
            old_q     = self.Q[prev_state][prev_act]
            self.Q[prev_state][prev_act] += self.alpha * (td_target - old_q)
            print(f"[DEBUG]   Q[{prev_state}][{prev_act}] updated from {old_q:.4f}"
                f" to {self.Q[prev_state][prev_act]:.4f}")



        # 5) book-keeping for next round
        task['_prev_state']  = next_state
        task['_prev_action'] = curr_act

        # decay epsilon after each round
        if curr_round >= 1:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        # logs
        self.reward_log.append({
            'round':       curr_round,
            'vehicle_id':  vid,
            'reward':      reward,
            'action':      curr_act,
            'prev_action': prev_act
        })
        self.epsilon_log.append({
            'round':   curr_round,
            'epsilon': self.epsilon
        })

   
    def save(self):
        with open(self.Q_FILENAME, "wb") as f:
            pickle.dump(self.Q, f)

    def save_logs(self, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        if self.reward_log:
            pd.DataFrame(self.reward_log).to_csv(f"{output_dir}/rewards.csv", index=False)
        if self.epsilon_log:
            pd.DataFrame(self.epsilon_log).to_csv(f"{output_dir}/epsilon_decay.csv", index=False)





    def train_until_perfect(self,
                             simulator_class,
                             max_epochs: int = 500,
                             tolerance: int = 0,
                             streak: int = 1):
        """
        Runs full Simulator.train_mode epochs until
        negative-reward count ≤ tolerance for `streak` in a row.

        simulator_class must be your core Simulator,
        so we can fresh-instantiate it each epoch.
        """
        import pandas as pd

        perfect = 0
        for epoch in range(1, max_epochs+1):
            # reset logs for this epoch
            self.reward_log.clear()
            self.epsilon_log.clear()

            # fresh sim, fresh state
            sim = simulator_class(self)

            # run one full training pass
            sim.run(train_mode=True)

            # count mistakes (negative rewards)
            df    = pd.DataFrame(self.reward_log)
            wrong = int((df['reward'] < 0).sum())
            print(f"Epoch {epoch:3d} → wrong decisions = {wrong}, ε = {self.epsilon:.3f}")

            # track perfect-run streak
            if wrong <= tolerance:
                perfect += 1
                if perfect >= streak:
                    print(f"✅ Reached {streak} perfect run(s) at epoch {epoch}.")
                    break
            else:
                perfect = 0

        else:
            print(f"⚠️  Hit {max_epochs} epochs without a zero-error streak.")

        # save out final Q-table and logs
        self.save()
        self.save_logs()
