# DriveNetSim

DriveNetSim is an Artificial Intelligence-based vehicular edge computing simulator designed for mobility aware task offloading, intelligent handover management, collaborative vehicular execution, and multi tier resource orchestration in dynamic vehicular environments.

The simulator models vehicle mobility, wireless communication, base station association, queue dynamics, resource utilization, workload generation, and distributed task execution across Vehicular Computing (VC), Mobile Edge Computing (MEC), Metro Edge (METRO), and Cloud Computing (CLOUD) layers.

DriveNetSim is developed for research in:

* Vehicular Edge Computing
* Intelligent Transportation Systems
* Reinforcement Learning based Scheduling
* Mobility Aware Offloading
* Queue Aware Resource Management
* Collaborative Vehicular Computing
* Stability Aware Decision Systems
* Multi Tier Edge Cloud Architectures

---

# System Architecture


<img width="3308" height="1424" alt="multi_tier_architecture" src="https://github.com/user-attachments/assets/2c4edc8e-b672-4159-8772-e3b0106378e3" />


The simulator implements a hierarchical multi tier computing architecture consisting of:


Vehicles perform local sensing, onboard processing, and delay sensitive execution.

## Tier 2 — Mobile Edge Computing (MEC)

Base stations provide wireless access and edge computation services with reduced communication delay.

## Collaborative Vehicular Execution

Vehicles within the same coverage area can cooperatively execute tasks through infrastructure assisted collaboration.

## Tier 3 — Metro Edge Computing (METRO)

Regional edge nodes provide higher computational capacity and workload aggregation.

## Tier 4 — Cloud Computing (CLOUD)

Centralized cloud infrastructure supports large scale processing and long term service execution.

The architecture supports:

* Vehicle to Infrastructure communication
* Vehicle to Vehicle collaboration
* Multi hop execution paths
* Mobility aware connectivity
* Dynamic tier migration
* Delay aware scheduling
* Queue aware resource management

---

# Simulation Workflow


<img width="1480" height="3388" alt="simulation_flow" src="https://github.com/user-attachments/assets/42c01a7f-7fcc-42a3-9c64-b054a1ca1cbe" />

Each simulation round follows a complete vehicular edge execution pipeline:

1. Configuration loading
2. Vehicle initialization
3. Mobility update
4. Base station association update
5. Workload generation
6. AI based handover decision
7. Queue state update
8. Feasibility validation
9. Reinforcement learning based offloading decision
10. Execution tier allocation
11. Delay and utilization estimation
12. Reward and regret computation
13. CSV logging and visualization

The framework supports continuous iterative learning across simulation rounds.

---

# Core Features

## Vehicle Mobility Modeling

* Dynamic vehicle movement
* Circular and road based mobility models
* Base station attachment
* Mobility aware connectivity tracking
* Handover simulation
* Coverage estimation
* Signal strength monitoring

---

## Multi Tier Task Offloading

Tasks can be executed on:

* Local vehicles
* MEC servers
* Collaborative vehicles
* Metro edge nodes
* Cloud servers

The scheduler dynamically selects the execution tier according to:

* Resource availability
* Queue state
* Communication delay
* Mobility constraints
* Service priority
* Utilization state
* Execution feasibility

---

## Reinforcement Learning Based Scheduling

DriveNetSim supports:

* Q Learning
* Deep Q Networks (DQN)
* PPO based scheduling
* Oracle guided evaluation
* Regret analysis
* Stability aware reward shaping

The RL scheduler learns adaptive offloading policies under dynamic network and workload conditions.

---

## Queue and Priority Aware Scheduling

The simulator supports heterogeneous traffic classes:

* URLLC
* mMTC
* eMBB

Features include:

* Parallel priority queues
* Queue delay estimation
* Starvation tracking
* Waiting time analysis
* Congestion aware scheduling
* Queue stability monitoring

---

## Collaborative Vehicular Computing

DriveNetSim models infrastructure assisted vehicle collaboration.

The collaboration pipeline includes:

* Source vehicle selection
* Base station relay coordination
* Candidate discovery
* Target vehicle allocation
* Multi hop task forwarding

This enables cooperative workload sharing within the same coverage region.

---

## Delay and Communication Modeling

The simulator models:

* Wireless transmission delay
* Wired propagation delay
* Queue waiting delay
* Processing delay
* End to end delay
* Handover delay
* Communication overhead

Both wireless and wired communication paths are supported.

---

## Resource Modeling

Each computing tier maintains:

* CPU utilization
* Memory utilization
* Queue state
* Task execution statistics
* Server availability
* Workload capacity

Dynamic resource fluctuations are continuously monitored during execution.

---

# Live Visualization Interface

DriveNetSim includes a real time graphical visualization framework.

<img width="1593" height="727" alt="experimentation_priority" src="https://github.com/user-attachments/assets/40689e5e-9081-458a-a67b-96eddeea1753" />

The GUI supports:

* Vehicle mobility visualization
* Base station association tracking
* Active offloading path rendering
* Collaboration visualization
* Queue monitoring
* Tier utilization statistics
* Live workload monitoring
* Metro and cloud activity tracking

The visualization system enables interactive monitoring of large scale vehicular simulations.

---

# Experimentation Support

DriveNetSim supports detailed experimentation for:

* Delay constrained task execution
* Mobility aware scheduling
* Priority aware execution
* Congestion analysis
* Resource utilization analysis
* Stability evaluation
* Queue dynamics analysis
* Oracle policy comparison
* Regret diagnostics
* Handover behavior analysis

The simulator generates CSV based outputs for post processing and scientific visualization.

---

# Evaluation Metrics

The framework supports analysis of:

* End to end delay
* Waiting delay
* Queue buildup
* Resource utilization
* Handover frequency
* Offloading distribution
* Oracle alignment
* Decision regret
* Switching ratio
* Persistence length
* Task completion ratio
* Starvation probability
* Priority wise execution delay

---


<img width="3648" height="2684" alt="class_diagram" src="https://github.com/user-attachments/assets/9bc2a1ef-ff13-42f7-8fd4-d3e953f498aa" />


# Repository Structure

```text
DriveNetSim/
│
├── application/
│   └── application.py
│
├── core/
│   ├── simulator.py
│   ├── reporter.py
│   └── visualization/
│
├── config/
│   ├── config.py
│   └── metrics.py
│
├── entities/
│   ├── server.py
│   ├── vehicle.py
│   └── task.py
│
├── mobility/
│   ├── manager.py
│   ├── mobility_model.py
│   └── association.py
│
├── scheduler/
│   ├── ql_scheduler.py
│   ├── dqn_scheduler.py
│   ├── collaborative_scheduler.py
│   └── reward_models/
│
├── workload/
│   ├── generator.py
│   └── priority_generator.py
│
├── live/
│   └── vehicular_map.py
│
├── trained_model/
│   ├── q_table.pkl
│   ├── actor.pth
│   └── critic.pth
│
├── output/
│   ├── offloading_results.csv
│   ├── handover_results.csv
│   ├── training_logs.csv
│   └── utilization_logs.csv
│
└── figures/
```

---

# Quick Start

## 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies include:

```bash
pip install numpy pandas matplotlib torch networkx pygame
```

---

## 3. Run the Simulator

```bash
python -m application.application
```

---

# Reproducibility

For reproducible experiments:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

# Related Publications

### Q Learning Based Mobility Aware Offloading for Multi Tier Vehicular Edge and Cloud Networks

Afzal Badshah, Ali Daud, and Umar Farooq.
Proceedings of the 7th International Conference on Advancements in Computational Sciences (ICACS), IEEE, 2026.

### Mobility Aware Q Learning for Workload Offloading in Vehicular Edge Cloud Environment

Afzal Badshah, Abdulrahman Ahmed Gharawi, Mona Eisa, Nada Alzaben, Saud Yonbawi, and Ali Daud.
*Pervasive and Mobile Computing*, Elsevier, 2026.

### Decision Stability and Regret Diagnostics for Reinforcement Learning Based Handover in Vehicular Mobility

Afzal Badshah, Ahmed S. Alzahrani, Mohammad D. Alahmadi, Sakher Ghanem, Raed Alsini, and Ali Daud.
*IET Communications*, Wiley, Vol. 20, No. 1, 2026.

---

# License

This project is released for academic and research purposes.

---

# Repository

[DriveNetSim GitHub Repository](https://github.com/afzalbadshah/DriveNetSim)
