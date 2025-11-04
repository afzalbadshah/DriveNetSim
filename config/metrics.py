# config/metrics.py
import math
import config.config as cfg

def calculate_transmission_delay(data_kb, bandwidth_mbps, server):
   return (data_kb * 8) / (bandwidth_mbps * 1000)  # delay in seconds
 



def wireless_delay(distance_m):
    """Delay for an EM hop (vehicle↔BS). Returns ms."""
    base_ms = distance_m / cfg.WIRELESS_PROP_SPEED * 1e3
    return (base_ms *8) * cfg.WIRELESS_DELAY_SCALE


def wired_delay(distance_m):
    """Delay for a fiber/copper hop (BS↔Cloud). Returns ms."""
    base_ms = distance_m / cfg.WIRED_PROP_SPEED * 1e3
    return base_ms * cfg.WIRED_DELAY_SCALE
    


def calculate_signal_strength(distance_m):
    n = cfg.SIGNAL['path_loss_exponent']
    PL0 = cfg.SIGNAL['reference_loss_db']
    d0 = cfg.SIGNAL['reference_distance']
    return cfg.SIGNAL['tx_power_dbm'] - (PL0 + 10*n*math.log10(distance_m/d0+1e-6))

def calculate_tx_cost(data_kb, tier):
    return data_kb * cfg.COST[tier]['tx_per_kb']

def calculate_proc_cost(proc_ms, tier):
    return proc_ms * cfg.COST[tier]['proc_per_ms']
