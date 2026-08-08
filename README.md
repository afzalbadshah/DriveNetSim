<div align="center">

# DriveNetSim

### A powerful Python simulator for mobility-aware vehicular communication, intelligent handover, and multi-tier task offloading

[![Latest Release](https://img.shields.io/github/v/release/afzalbadshah/DriveNetSim?display_name=tag&label=Latest%20Release&color=brightgreen)](https://github.com/afzalbadshah/DriveNetSim/releases/latest)
[![Release Date](https://img.shields.io/github/release-date/afzalbadshah/DriveNetSim)](https://github.com/afzalbadshah/DriveNetSim/releases)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Schedulers](https://img.shields.io/badge/Schedulers-Random%20%7C%20Q--Learning%20%7C%20DQN%20%7C%20PPO%20%7C%20A2C-8A2BE2)

</div>

> [!IMPORTANT]
> ## Get the latest version
>
> Download the most recently published DriveNetSim package from the
> **[Latest GitHub Release ->](https://github.com/afzalbadshah/DriveNetSim/releases/latest)**
>
> DriveNetSim releases represent different research tracks. For exact reproducibility, use the release cited by the corresponding publication or experiment.

## Overview

DriveNetSim is a modular, event-driven Python simulation framework for studying vehicular computing and communication under mobility, changing wireless conditions, dynamic infrastructure load, heterogeneous workloads, and multi-tier execution constraints.

It brings the following processes into one reproducible simulation pipeline:

- Vehicular mobility and continuous road movement
- Wireless communication, RSSI, SNR, path loss, fading, and connectivity
- Base-station association and mobility-aware handover
- Queue-aware and priority-aware task scheduling
- Vehicular, edge, collaborative, metro-edge, and cloud execution
- Feasibility-constrained task offloading
- AI-driven and heuristic scheduling
- Oracle comparison, regret analysis, and decision-stability diagnostics
- Live mobility, offloading, collaboration, and queue visualization
- Structured CSV logging for post-processing and scientific evaluation

DriveNetSim is designed for research in vehicular edge computing, intelligent transportation systems, V2X communication, mobile edge computing, resource-aware handover, reinforcement learning, queue-aware orchestration, collaborative vehicular computing, and next-generation 5G/6G vehicular networks.

## Why DriveNetSim?

Many simulators model mobility, networking, or distributed computing as separate concerns. DriveNetSim connects them within a common runtime so that a scheduling decision is evaluated against the conditions that actually constrain it:

- Is the wireless link usable?
- Will the vehicle remain connected long enough?
- Is the selected server feasible?
- How much work is already waiting in its queues?
- Does the decision satisfy the task's service class and deadline?
- Would another valid action provide higher utility?
- Does the learned policy agree with an oracle reference?
- How much regret is introduced by a suboptimal decision?

This integrated design makes DriveNetSim suitable both as a vehicular simulator and as a controlled benchmarking environment for intelligent scheduling policies.

## Key capabilities

### Mobility and BS association

- Continuous vehicle movement on configurable road topologies
- Circular-road mobility for uninterrupted BS transitions
- Vehicle position, speed, serving BS, coverage, and dwell-time tracking
- Dynamic BS association
- AI-driven and rule-based handover support
- Handover delay, interruption, failure, and ping-pong diagnostics
- Mobility-feasibility checks before task execution

### Wireless communication

- Distance-dependent path loss
- RSSI and SNR estimation
- Noise and interference tracking
- Wireless and wired propagation delay
- Configurable communication thresholds
- Rayleigh, Rician, and Nakagami fading abstractions
- Link-quality-aware BS and execution-tier selection

### Queue-aware and priority-aware execution

DriveNetSim models independent workload classes and runtime queue pressure. Queue state can influence action masking, reward calculation, waiting-delay estimation, and tier selection.

Supported 3GPP-oriented service classes include:

| Class | Full name | Typical behavior |
|---|---|---|
| **URLLC** | Ultra-Reliable Low-Latency Communications | Strict deadlines, safety-critical and latency-sensitive processing |
| **mMTC** | Massive Machine-Type Communications | High-volume sensing and telemetry workloads |
| **eMBB** | Enhanced Mobile Broadband | Bandwidth-intensive data and infotainment workloads |

Queue instrumentation includes:

- Queue length by service class and execution tier
- Queued workload
- Waiting and service time
- Deadline and starvation indicators
- Congestion and utilization state
- Priority-sensitive scheduling and admission control

### Multi-tier task offloading

Tasks can be processed across five logical destinations:

| Tier | Role |
|---|---|
| **VE / VC** | Local onboard vehicular execution |
| **BS / MEC** | Execution at the serving base-station edge server |
| **Collaboration** | BS-mediated execution by a feasible helper vehicle |
| **ME** | Regional or metro-edge execution |
| **CC** | Remote cloud execution and fallback |

The selected tier can depend on mobility, wireless quality, data size, CPU and memory demand, queue state, waiting time, service deadline, infrastructure utilization, cost, energy proxy, and action feasibility.

## System architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c4edc8e-b672-4159-8772-e3b0106378e3" alt="DriveNetSim multi-tier architecture" width="100%">
</p>

The base station acts both as a wireless access point and as a gateway toward collaboration, metro-edge, and cloud resources. For collaborative execution, it can discover a helper vehicle within the same coverage region and relay the task between the source and target vehicles.

A collaborative action is admitted only when its communication time, helper-queue delay, processing time, remaining dwell time, and resource state satisfy the configured feasibility constraints.

## Simulation workflow

<p align="center">
  <img src="https://github.com/user-attachments/assets/42c01a7f-7fcc-42a3-9c64-b054a1ca1cbe" alt="DriveNetSim simulation workflow" width="55%">
</p>

Each simulation round follows a controlled execution sequence:

1. Load configuration and experiment parameters.
2. Initialize vehicles, infrastructure, queues, and scheduling state.
3. Update vehicle positions and mobility state.
4. Refresh BS association and wireless measurements.
5. Generate class-aware vehicular workloads.
6. Select or update the serving BS.
7. Estimate queue and waiting-time state.
8. Construct the scheduler state.
9. Apply communication, mobility, queue, resource, and priority feasibility checks.
10. Select an offloading action.
11. Allocate the task to VE, BS, Collaboration, ME, or CC.
12. Estimate transmission, propagation, queue, processing, and handover delay.
13. Calculate reward, oracle agreement, and regret.
14. Write structured outputs and update visualizations.
15. Advance to the next round.

## Scheduling models

DriveNetSim's unified evaluation design supports controlled comparison of five scheduling families under identical mobility, communication, workload, queue, and resource conditions:

| Scheduler | Purpose |
|---|---|
| **Random** | Heuristic baseline using random feasible-action selection |
| **Q-Learning** | Tabular value learning for discrete state and action spaces |
| **DQN** | Neural action-value estimation with replay and feasibility masking |
| **PPO** | Actor-critic policy optimization with constrained action selection |
| **A2C** | Advantage actor-critic learning for policy and value estimation |

The oracle is not an online deployment model. It is an offline reference that evaluates feasible actions using complete instantaneous state information and provides an upper-bound decision for comparison.

> [!NOTE]
> The research manuscript describes the unified five-model benchmark. The public repository currently packages several research variants separately:
>
> - `main` contains the baseline Q-learning-oriented source.
> - The latest release focuses on actor-critic BS handover.
> - V1.9 contains queue-aware collaborative DQN components.
> - MARL and logistics variants are distributed through their respective releases.
>
> When reproducing a result, use the exact tagged release associated with that experiment. A consolidated release should expose all five schedulers through one configuration-driven selector.

### Selecting a scheduler

Scheduler selection is release-specific:

- The main branch selects its scheduler in `application/application.py`.
- The latest BS-handover release imports `scheduler.dacscheduler.Scheduler`.
- The V1.9 collaboration package provides `scheduler/collaborative_dqscheduler.py`.
- Advanced release configuration is centralized in `config/config.py`.

After selecting or importing the required scheduler, use the same simulator entry point:

```bash
python -m application.application
```

For fair comparisons, keep topology, mobility, workload, seed, QoS, queue policy, and resource parameters unchanged between scheduler runs.

## Feasibility masking

DriveNetSim restricts each task to actions that are valid at the current simulation state.

Conceptually, an action is valid only when:

```text
link feasible
AND mobility feasible
AND resource feasible
AND queue feasible
AND priority/QoS feasible
```

The mask prevents the scheduler from selecting tiers that cannot maintain connectivity, finish before a mobility transition, satisfy resource demand, accept additional queue pressure, or meet the task's service constraints.

For collaborative execution, the simulator additionally checks:

- Source and helper BS association
- Helper availability
- Communication continuity
- Remaining dwell time
- Source-to-BS and BS-to-helper transfer delay
- Helper queue and processing delay
- CPU, memory, and storage availability

## Oracle and regret diagnostics

DriveNetSim supports decision-level evaluation rather than relying only on average reward.

For each scheduling decision, the reporter can record:

- Scheduler-selected action
- Oracle action
- Oracle agreement indicator
- Regret magnitude
- Regret occurrence rate
- Action margin or confidence
- Handover and tier-switching behavior
- Mobility and resource conditions at decision time

Regret measures the utility gap between the oracle-selected feasible action and the scheduler-selected action. These diagnostics help distinguish a stable policy from one that achieves a similar average reward while producing unreliable individual decisions.

## Live visualization

DriveNetSim includes GUI-enabled releases for observing mobility, infrastructure state, active offloading paths, collaborative execution, and queue pressure during runtime.

<p align="center">
  <img src="https://github.com/user-attachments/assets/40689e5e-9081-458a-a67b-96eddeea1753" alt="DriveNetSim queue and priority runtime view" width="100%">
</p>

The interface can display:

- Vehicles and BS attachment
- Current mobility and handover state
- Active VE, BS, Collaboration, ME, and CC paths
- Source-to-BS and BS-to-helper collaboration hops
- CPU and memory utilization
- Queue length and active tasks
- Metro-edge and cloud activity
- Candidate-helper discovery
- Per-task transfer information

## Software architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/9bc2a1ef-ff13-42f7-8fd4-d3e953f498aa" alt="DriveNetSim class-level software architecture" width="100%">
</p>

The simulator separates configuration, mobility, scheduling, execution, reporting, and visualization so that a scheduler can be replaced without rewriting the core simulation loop.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/afzalbadshah/DriveNetSim.git
cd DriveNetSim
```

To use a versioned research artifact instead, download and extract the appropriate package from [GitHub Releases](https://github.com/afzalbadshah/DriveNetSim/releases).

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core Python dependencies are:

- NumPy
- pandas
- PyTorch
- Matplotlib

GUI-enabled releases also use Tkinter. Tkinter is normally included with Windows and macOS Python distributions. On Debian or Ubuntu, install it with:

```bash
sudo apt-get install python3-tk
```

## Quick start

Run the simulator from the repository root:

```bash
python -m application.application
```

The checked-in application entry point runs in evaluation mode by default.

To train a scheduler:

1. Open `application/application.py`.
2. Set the script's training flag to `True`.
3. Run the same command:

```bash
python -m application.application
```

Training and evaluation behavior varies by release. Model checkpoints and output folders should therefore be kept with their corresponding configuration and release tag.

## Configuration

The central configuration file is:

```text
config/config.py
```

Depending on the selected release, it controls:

| Category | Examples |
|---|---|
| Topology | Number of BSs, BS spacing, road length, metro nodes |
| Simulation | Number of rounds and round duration |
| Mobility | Vehicle speed, handover threshold, dwell margin |
| Wireless | Transmit power, path loss, fading, RSSI and SNR thresholds |
| Infrastructure | VE, BS/MEC, metro-edge, and cloud capacity |
| Workloads | Vehicle count, data-size distribution, CPU/memory/storage demand |
| QoS | URLLC, mMTC, and eMBB deadlines |
| Queueing | Queue policy, starvation threshold, preemption setting |
| Learning | Learning rate, discount factor, epsilon schedule, replay size |
| Feasibility | Utilization limits, mobility margins, strict action masking |
| Collaboration | Helper threshold, transfer rate, feasibility margin |
| Cost and energy | Transmission, processing, and energy coefficients |
| Logging | Training outputs and diagnostic controls |

Change one experimental factor at a time and preserve the full configuration used for each reported result.

## Outputs

DriveNetSim writes generated data under `output/` or release-specific output directories.

| Output | Description |
|---|---|
| `output/results.csv` | Combined task, mobility, signal, decision, delay, and utilization records |
| `output/handover_results.csv` | BS association, RSSI/SNR, dwell, handover, and mobility diagnostics |
| `output/offloading_results.csv` | Tier decision, mask, queue, delay, cost, utilization, and collaboration fields |
| `output/decision_quality.csv` | Chosen action, oracle action, regret, and decision-quality information |
| `output/rewards.csv` | Reward progression |
| `output/epsilon_decay.csv` | Exploration schedule |
| `output/action_distribution.csv` | Distribution of selected execution tiers |
| `output/loss_log.csv` or equivalent | Actor, critic, or DQN training loss |
| `output/*.png` | Training, regret, delay, utilization, and policy-quality plots |

Depending on the release, CSV rows can include:

- Round and vehicle identity
- Workload class, priority, size, and deadline
- Connected BS and handover state
- Vehicle position, speed, and remaining dwell time
- RSSI, SNR, noise, interference, and distance
- Selected and oracle actions
- Feasibility mask and valid-action count
- Queue length, queued workload, waiting time, and starvation state
- Transmission, propagation, processing, queue, handover, and end-to-end delay
- CPU, memory, and storage utilization
- Execution cost and energy proxy
- Regret and oracle agreement
- Collaboration checks, helper identity, timing budget, and rejection reason

The V1.9 reporter writes semicolon-separated CSV files for spreadsheet compatibility.

## Reproducibility

For reproducible research:

1. Use a tagged GitHub release rather than an unversioned archive.
2. Record the release tag and commit.
3. Preserve `config/config.py` with the results.
4. Fix Python, NumPy, and PyTorch random seeds.
5. Keep training and evaluation workload profiles separate.
6. Save scheduler checkpoints with their model and configuration.
7. Do not reuse checkpoints across incompatible state or action dimensions.
8. Retain raw CSV files before post-processing.
9. Report queue policy, QoS deadlines, topology, vehicle count, and simulation rounds.
10. Run all scheduler comparisons under identical conditions.

## Repository structure

The exact contents vary between research releases, but the simulator follows this package organization:

```text
DriveNetSim/
|-- application/
|   `-- application.py          # Run entry point in main and GUI-enabled releases
|-- config/
|   |-- config.py               # Topology, workload, QoS, queue, and learning settings
|   `-- metrics.py              # Signal, delay, cost, energy, and communication models
|-- core/
|   |-- simulator.py            # Event-driven simulation loop
|   `-- reporter.py             # CSV output and diagnostics
|-- entities/
|   |-- server.py               # Computing-tier resources
|   `-- server_queue.py         # Runtime queues in queue-aware releases
|-- mobility/
|   |-- manager.py              # Vehicle movement and BS association
|   `-- acmobility.py           # AI-based handover in applicable releases
|-- scheduler/
|   |-- ql_scheduler.py         # Q-learning baseline on main
|   |-- collaborative_dqscheduler.py
|   `-- scheduler_helper/
|       |-- vehicular_collaboration.py
|       `-- training_visualization_helper.py
|-- workload/
|   `-- generator.py            # Class-aware task generation
|-- live/
|   |-- vehicular_map.py        # Runtime GUI in GUI-enabled releases
|   `-- assets/
|-- trained_model/              # Generated model checkpoints
|-- output/                     # Generated CSV files and plots
|-- q_table.pkl                 # Baseline Q-table on main
|-- requirements.txt
`-- README.md
```

## Example experiments

### Scheduler comparison

Evaluate Random, Q-Learning, DQN, PPO, and A2C under the same topology, workload, queue, mobility, and seed configuration. Compare:

- Average reward
- Oracle-aligned decision rate
- Regret magnitude
- Tier allocation
- End-to-end and tail latency
- Queue length and waiting time
- Resource utilization

### Priority and queue analysis

Vary the URLLC, mMTC, and eMBB workload proportions and study:

- Waiting-time distributions
- P95 and P99 latency
- Starvation frequency
- Queue buildup by tier
- Priority-specific deadline violations

### Collaborative execution

Enable vehicular collaboration and evaluate:

- Helper discovery rate
- Collaboration feasibility
- Collaboration versus metro fallback
- Mobility-induced collaboration rejection
- Communication and processing budgets
- URLLC tail-latency improvement

### Mobility and handover

Change vehicle speed, BS spacing, fading, or dwell margins and measure:

- Handover frequency
- RSSI and SNR
- Connectivity continuity
- Ping-pong behavior
- Handover delay and failure
- Oracle agreement and regret

### Stress evaluation

Use an increasing workload or vehicle ramp to examine:

- Congestion onset
- Tier escalation
- Queue stability
- Resource saturation
- Cloud fallback
- Policy robustness

## Adding a custom scheduler

A custom scheduler should follow the interface expected by the selected release's `core/simulator.py`.

The common integration points are:

```python
set_train_mode(is_training)
select_server(task, servers, mobility)
save()
save_logs(...)
```

Learning-based releases may additionally call:

```python
calculate_reward(...)
update(...)
post_step_update(...)
```

Recommended integration procedure:

1. Add the implementation under `scheduler/`.
2. Use an existing scheduler from the same release as the interface reference.
3. Accept the task, server list, and mobility manager supplied by the simulator.
4. Construct only the state needed by the new policy.
5. Apply the simulator's feasibility mask before action selection.
6. Return a server or execution-tier action in the format expected by that release.
7. Implement training, checkpoint, and diagnostic hooks where required.
8. Import and instantiate the scheduler in `application/application.py`.
9. Keep the surrounding mobility, workload, and reporting pipeline unchanged.
10. Validate the scheduler against Random and oracle references under the same configuration.

## Releases

DriveNetSim tags represent different research tracks rather than one strictly sequential product line. Use GitHub's **Latest** marker for the most recently published package and the cited tag for paper reproduction.

| Release | Focus |
|---|---|
| [Latest release](https://github.com/afzalbadshah/DriveNetSim/releases/latest) | Most recently published DriveNetSim package |
| [V1.9](https://github.com/afzalbadshah/DriveNetSim/releases/tag/V1.9) | Queue-aware, collaborative, multi-tier task offloading; cited by the simulator manuscript |
| [V1.8](https://github.com/afzalbadshah/DriveNetSim/releases/tag/V1.8) | Collaboration-aware simulator and generated dataset |
| [V1.7 MARL](https://github.com/afzalbadshah/DriveNetSim/releases/tag/V1.7MARL) | Multi-agent handover and offloading |
| [V1.7 Logistics](https://github.com/afzalbadshah/DriveNetSim/releases/tag/V1.7_Logistic) | Class-aware smart-logistics and ITS workloads |
| [All releases](https://github.com/afzalbadshah/DriveNetSim/releases) | Complete release history and downloadable artifacts |

## Citation and publications

If DriveNetSim supports your research, please cite the simulator and the relevant methodology paper.

```bibtex
@software{badshah2026drivenetsim,
  author  = {Afzal Badshah},
  title   = {DriveNetSim: A Simulation Framework for Mobility-Aware
             and Learning-Based Vehicular Task Offloading},
  year    = {2026},
  version = {1.9},
  url     = {https://github.com/afzalbadshah/DriveNetSim}
}
```

### Related publications

1. A. Badshah, A. A. Gharawi, M. Eisa, N. Alzaben, S. Yonbawi, and A. Daud, "Mobility-aware Q-learning for workload offloading in vehicular edge-cloud environment," *Pervasive and Mobile Computing*, vol. 117, 102172, 2026. [https://doi.org/10.1016/j.pmcj.2026.102172](https://doi.org/10.1016/j.pmcj.2026.102172)

2. A. Badshah, A. S. Alzahrani, M. D. Alahmadi, S. Ghanem, R. Alsini, and A. Daud, "Decision-stability and regret diagnostics for reinforcement learning based handover in vehicular mobility," *IET Communications*, vol. 20, no. 1, e70159, 2026. [https://doi.org/10.1049/cmu2.70159](https://doi.org/10.1049/cmu2.70159)

3. A. Badshah, A. Daud, and U. Farooq, "Q-learning-based mobility-aware offloading for multi-tier vehicular edge and cloud networks," in *2026 7th International Conference on Advancements in Computational Sciences (ICACS)*, IEEE, pp. 1-6, 2026.

## Contributing

Contributions that improve simulator correctness, reproducibility, documentation, scheduling support, mobility realism, or evaluation tooling are welcome.

Suggested workflow:

1. Fork the repository.
2. Create a focused feature or fix branch.
3. Keep scheduler-specific logic inside `scheduler/`.
4. Avoid changing the simulation environment solely to favor one model.
5. Document new configuration options.
6. Include a reproducible example and expected outputs.
7. Verify that existing entry points still run.
8. Open a pull request describing the motivation, changes, and validation.

Bug reports should include the release tag, Python version, configuration, command used, traceback, and a minimal reproduction.

## License

DriveNetSim is currently distributed for academic and research use. A standalone `LICENSE` file is not yet included in the repository.

Until an explicit license is added, users should contact the author before redistribution, commercial use, or incorporation into another project. Public source availability should not be interpreted as an OSI-approved open-source license.

## Contact

**Afzal Badshah**<br>
Department of Software Engineering<br>
University of Sargodha, Pakistan

- GitHub: [@afzalbadshah](https://github.com/afzalbadshah)
- Website: [https://afzalbadshah.com/](https://afzalbadshah.com/)
- Email: [afzalbadshahphd@gmail.com](mailto:afzalbadshahphd@gmail.com)

---

<div align="center">

If DriveNetSim is useful in your work, please cite the relevant publication and consider starring the repository.

**[Download the latest release](https://github.com/afzalbadshah/DriveNetSim/releases/latest)** Â·
**[View all releases](https://github.com/afzalbadshah/DriveNetSim/releases)** Â·
**[Report an issue](https://github.com/afzalbadshah/DriveNetSim/issues)**

</div>
