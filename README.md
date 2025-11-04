# DriveNetSim

Reinforcement-learning–driven vehicular edge simulator with base-station mobility and task offloading.

## Features
- Vehicle mobility with circular road model and BS connectivity
- Signal strength & delay models (wireless/wired/processing)
- RL scheduler (Q-table based) + cost metrics & logging

## Quick start

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run a simulation
```bash
python -m application.application
```

### Folder map
- `application/`: entrypoint and small glue code
- `core/`: simulator loop and reporting
- `config/`: parameters and metrics
- `entities/`: server/resource model
- `mobility/`: vehicle positions & BS association
- `scheduler/`: Q-learning scheduler (`ql_scheduler.py`) and variants
- `workload/`: synthetic workload generator

### Q-table
The scheduler loads `q_table.pkl` if present. Remove it to start fresh, or let the code create/update it.

## License
MIT (or your preferred license).
