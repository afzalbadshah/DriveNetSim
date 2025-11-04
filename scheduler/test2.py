import os
import pickle
import random
import numpy as np
import pandas as pd
import config as cfg
import time
import csv

class AIScheduler:
    Q_FILENAME = "q_table.pkl"

    def __init__(self):
        self.train_mode = False
        self.Q = self._load_q_table()
        self.servers = None
        self.vehicle_bs = {}
        self.last_action = {}

       
        epsilon=cfg.EPSILON_START
        epsilon_min=cfg.EPSILON_MIN
        epsilon_decay=cfg.EPSILON_DECAY
        
        alpha = cfg.LEARNING_RATE
        gamma = cfg.DISCOUNT_FACTOR
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.min_epsilon = epsilon_min
        self.decay = epsilon_decay


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
        ve = self._get_server(self.servers, "VE", vehicle_id=vid)
        bs_idx, signal = mobility.connected_bs[f"VE_{vid}"]
        bs = self._get_server(self.servers, "BS", bs_id=bs_idx)

        #print(f"[DEBUG] VE {ve.id} CPU: {ve.get_cpu_util():.1f}%, MEM: {ve.get_mem_util():.1f}% BS {bs.id} CPU: {bs.get_cpu_util():.1f}%, MEM: {bs.get_mem_util():.1f}% ")
        #time.sleep(2)  # for better debug output readability   

        # Normalize raw util (0–1 or 0–100) into 0–100%
        def pct(u):
            return u * 100.0 if u < 1.0 else u

        ve_cpu_pct = pct(ve.get_cpu_util())
        ve_mem_pct = pct(ve.get_mem_util())
        bs_cpu_pct = pct(bs.get_cpu_util())
        bs_mem_pct = pct(bs.get_mem_util())

        # Bucket into 5%-bins
        ve_cpu = int(min(ve_cpu_pct, 100) // 5)
        ve_mem = int(min(ve_mem_pct, 100) // 5)
        bs_cpu = int(min(bs_cpu_pct, 100) // 5)
        bs_mem = int(min(bs_mem_pct, 100) // 5)

        # Signal comes in 0–1 or 0–100; store raw percentage
        bs_sig = pct(signal)

        priority = task.get("priority", 1)
        return (priority, ve_cpu, ve_mem, bs_sig)


    def _init_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(3)

    def _to_pct(self, u):
        return u * 100.0 if u < 1.0 else u


    def calculate_reward(self, ve, bs, cl, selected, sig, task, ve_cpu, ve_mem, bs_cpu, bs_mem):
        priority = task.get("priority", 1)
        signal_bonus = 0.5 if sig >= cfg.BS_SIG_THRESHOLD else -0.5
        
        raw = 1.0

        ve_under = (ve_cpu < 50 and ve_mem < 50)
        ve_over = (ve_cpu >= 50 or ve_mem >= 50)

     
        if priority in (2, 3) and selected == ve and ve_over:
            raw = -10.0  

        # 2) Base rewards by priority
        elif priority == 1:
            if selected == ve and ve.can_allocate(task):
                raw = 10.0 
            elif selected == bs and bs.can_allocate(task) and (ve_cpu > 80 or ve_mem > 80) and sig >= cfg.BS_SIG_THRESHOLD:
                raw = 3.0 + signal_bonus 
           

        elif priority == 2:
            if selected == ve and ve.can_allocate(task) and ve_under:
                raw = 7  
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                raw = 3 + signal_bonus
            #elif selected in (bs, cl) and (ve_cpu < 50 and ve_mem < 50 ):
                #raw = -10  
           

        elif priority == 3:
            if selected == ve and ve.can_allocate(task) and (ve_cpu < 50 and ve_mem < 50):
                raw = 6  
            elif selected == bs and bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD:
                raw = 3 + signal_bonus  
            elif selected == cl:
                raw = 3  
            #elif selected in (bs, cl) and (ve_cpu < 50 or ve_mem < 50 ):
                #raw = -10
        else:
            if selected != ve and ve.can_allocate(task) and ve_under:
             raw = -2.0
            elif selected == cl and bs.can_allocate(task) and bs_cpu < 70 and bs_mem < 70 and sig >= cfg.BS_SIG_THRESHOLD:
             raw = -2.0


        return raw / 5.0   

        # 3) Reduced skip-penalties (were -2 → now -1)
        #if selected != ve and ve.can_allocate(task) and (ve_cpu < 70 and ve_mem < 70):
            #raw -= 3.0  # Should have used VE

        #if selected == cl and bs.can_allocate(task) and (bs_cpu < 70 or bs_mem < 70) and sig >= cfg.BS_SIG_THRESHOLD:
            #raw -= 3.0  # Should have used BS

        # 4) Normalize into roughly [-1..+1]
   

    def select_server(self, task, servers, mobility):
        # stash for update()
        self.servers = servers

        vid   = task['vehicle_id']
        # build your state key (with bucketed CPU/mem!)
        state = self._get_state(task, mobility)
        self._init_state(state)

        task['_prev_state']  = state
        task['_prev_action'] = task.get('_last_action', -1)

        # fetch VE / BS / CL
        ve  = self._get_server(servers, "VE", vehicle_id=vid)
        bs_idx, sig = mobility.connected_bs[f"VE_{vid}"]
        bs  = self._get_server(servers, "BS", bs_id=bs_idx)
        cl  = self._get_server(servers, "CL")

        # === NEW: grab *raw* % utils ===
        ve_cpu_raw = self._to_pct(ve.get_cpu_util())
        ve_mem_raw = self._to_pct(ve.get_mem_util())
        bs_cpu_raw = self._to_pct(bs.get_cpu_util())
        bs_mem_raw = self._to_pct(bs.get_mem_util())

        per_task_cpu_pct = task['cpu_demand'] / ve.cpu * 100
        per_task_mem_pct = task['memory_demand'] / ve.memory_mb * 100

        ve_cpu_raw = ve_cpu_raw + per_task_cpu_pct
        ve_mem_raw = ve_mem_raw + per_task_mem_pct

        #print(f"[DEBUG] VE raw CPU: {ve_cpu_raw:.1f}%, MEM: {ve_mem_raw:.1f}%")

        # feasibility mask
        priority = task["priority"]
        VE_OVERLOAD_THRESHOLD = 50.0
        # use raw % here, *not* your 5%-bin
        ve_overloaded = (priority in (2, 3)) and (
            ve_cpu_raw >= VE_OVERLOAD_THRESHOLD or
            ve_mem_raw >= VE_OVERLOAD_THRESHOLD
        )


        #print(f"[Debug]: VE Overloaded: {ve_overloaded}, ")
        #time.sleep(2)  # for better debug output readability

        feasible = [
            ve.can_allocate(task) and not ve_overloaded,
            bs.can_allocate(task) and sig >= cfg.BS_SIG_THRESHOLD,
            cl.can_allocate(task)
        ]
        if not any(feasible):
            feasible = [False, False, True]

        #print(f"[DEBUG] Feasible actions: {feasible} (VE={ve.id}, BS={bs.id}, CL={cl.id})")

        # ε-greedy
        if self.train_mode and random.random() < self.epsilon:
            action = random.choice([i for i,ok in enumerate(feasible) if ok])
        else:
            qvals   = self.Q[state]
            masked  = [q if feasible[i] else -np.inf
                       for i,q in enumerate(qvals)]
            action  = int(np.argmax(masked))

        task['_last_action'] = action
        selected = [ve, bs, cl][action]

        # compute reward (pass raw % so penalties line up)
        reward = self.calculate_reward(
            ve, bs, cl, selected, sig, task,
            ve_cpu_raw, ve_mem_raw, bs_cpu_raw, bs_mem_raw
        )
        if self.train_mode:
            self.update(reward, task, mobility)

        self.last_action[vid] = action
        if action == 1:
            self.vehicle_bs[vid] = selected

        
#------------------------------------------------------------------
        header = ['round','vehicle','priority',
              've_cpu_raw','ve_mem_raw','bs_cpu_raw','bs_mem_raw',
              've_overloaded','feas_ve','feas_bs','feas_cl','selected']
        row    = [ task.get('round', -1),
               vid,
               priority,
               ve_cpu_raw, ve_mem_raw,
               bs_cpu_raw, bs_mem_raw,
               int(ve_overloaded),
               int(feasible[0]), int(feasible[1]), int(feasible[2]),
               action ]

        # write (append) to debug.csv
        file_existed = os.path.isfile('debug.csv')
        with open('debug.csv','a', newline='') as f:
            w = csv.writer(f)
            if not file_existed:
                w.writerow(header)
            w.writerow(row)
#------------------------------------------------------------------



        return selected


    def update(self, reward, task, mobility):
        # only learn in train mode
        if not self.train_mode:
            return

        prev_act   = task.get('_prev_action', -1)
        curr_act   = task.get('_last_action', -1)
        prev_state = task.get('_prev_state')
        curr_round = task.get('round', -1)

        #print(f"[DEBUG] update(): reward={reward}, prev_state={prev_state}, "
              #f"prev_act={prev_act}, curr_act={curr_act}")

        # 1) switching penalty
        if prev_act >= 0 and curr_act != prev_act:
            reward -= 2
            #print(f"[DEBUG]   switching penalty applied, new reward={reward}")

        # 2) simulate allocation on chosen server
        ve   = self._get_server(self.servers, "VE", vehicle_id=task['vehicle_id'])
        bs_i, _ = mobility.connected_bs[f"VE_{task['vehicle_id']}"]
        bs   = self._get_server(self.servers, "BS", bs_id=bs_i)
        cl   = self._get_server(self.servers, "CL")
        chosen = [ve, bs, cl][curr_act]
        if chosen.can_allocate(task):
            chosen.allocate_until_full(task, curr_round, util_threshold=1.0)

        # 3) compute next state
        next_state = self._get_state(task, mobility)
        self._init_state(next_state)
        #print(f"[DEBUG]   next_state={next_state}, Q[next_state]={self.Q[next_state]}")

        # 4) Q-learning update *only* if we had a real prev_act
        if prev_act >= 0:
            best_next = np.max(self.Q[next_state])
            td_target = reward + self.gamma * best_next
            old_q     = self.Q[prev_state][prev_act]
            self.Q[prev_state][prev_act] += self.alpha * (td_target - old_q)
            #print(f"[DEBUG]   Q[{prev_state}][{prev_act}] updated "
                  #f"from {old_q:.4f} to {self.Q[prev_state][prev_act]:.4f}")

        # 5) shift bookkeeping for next call
        task['_prev_state']  = next_state
        task['_prev_action'] = curr_act

        # 6) decay ε
        if curr_round >= 1:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        # 7) logging
        self.reward_log.append({
            'round':       curr_round,
            'vehicle_id':  task['vehicle_id'],
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


