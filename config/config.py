# config/config.py

#NUM_VEHICLES = 50
NUM_BASE_STATIONS = 10
DISTANCE_BETWEEN_BS = 5000
ROAD_LENGTH = NUM_BASE_STATIONS * DISTANCE_BETWEEN_BS
SIMULATION_ROUNDS = 1000

WIRELESS_PROP_SPEED = 3e8         
WIRED_PROP_SPEED     = 2e8        

VE_PROP_DELAY_MS    = 1.0
BS_CLOUD_DISTANCE_M = 20_000  

WIRELESS_DELAY_SCALE = 1000.0  
WIRED_DELAY_SCALE     = 100.0  

HANDOVER_PENALTY = 50.0
OVERLOAD_PENALTY = 100.0
LOAD_MAX = 70.0

SIGNAL_THRESH_HOLD = -80.0
SIGNAL_THRESH_GO   = -65.0
MARGIN_DB         = 3.0 
LOAD_MAX          = 70.0
VE_UTIL_THRESHOLD = 80.0
BS_UTIL_THRESHOLD = 80.0
BS_SIG_THRESHOLD  = -75.0


# exploration
EPSILON_START   = 1.0       # initial ϵ for ε-greedy
EPSILON_DECAY   = 0.999     # per-step decay of ϵ
EPSILON_MIN     = 0.1       # floor on ϵ

# learning
LEARNING_RATE   = 0.1       # α
DISCOUNT_FACTOR = 0.9       # γ




VEHICULAR_EDGE = {
    "cpu": 500_000,                 # MIPS
    "memory_mb": 16000,             # MB
    "storage_gb": 500,              # GB
    "bandwidth_mbps": 1000,         # Mbps
    "propagation_delay_ms": 1.0,    # ms    
    "propagation_delay_ms": 1.0     # ms 
}

BS_EDGE = {
    "cpu": 6_400_00,
    "memory_mb": 25600,
    "storage_gb": 1000,
    "bandwidth_mbps": 500,
    "propagation_delay_ms": 5.0,
    "handover_delay_ms": 2.0
}

CLOUD = {
    "cpu": 10_080_0000,
    "memory_mb": 2000000,
    "storage_gb": 4000,
    "bandwidth_mbps": 2000,
    "propagation_delay_ms": 20.0
}

WORKLOAD = {
    "cpu_demand": 100,
    "memory_demand": 100,
    "storage_demand": 50,
    "data_per_device_kb_range": [100, 100000],
    "data_increment":50,
    "num_vehicles_range": [10, 10],
    "vehicle_increment": 2,
    "cpu_per_kb":    10, 
    "mem_per_kb":    0.2,

    "priority_distribution": {
        1: 0.2,
        2: 0.5,
        3: 0.3
    }
}

MOBILITY_CONFIG = {
    "vehicle_speed_mean": 15,
    "handover_threshold": 50
}

BS_POSITIONS = [i * DISTANCE_BETWEEN_BS for i in range(NUM_BASE_STATIONS)]

COST = {
    "vehicular_edge": {"tx_per_kb": 0.00001, "proc_per_ms": 0.00003},
    "bs_edge":        {"tx_per_kb": 0.00005, "proc_per_ms": 0.00004},
    "cloud":          {"tx_per_kb": 0.00008, "proc_per_ms": 0.00005}
}

SIGNAL = {
    "tx_power_dbm": 20,
    "path_loss_exponent": 2.0,
    "reference_distance": 1.0,
    "reference_loss_db": 30
}
