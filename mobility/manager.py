import random
import config.config as cfg
from config.metrics import calculate_signal_strength


class MobilityManager:
    """
    1) initialize_positions(): evenly space vehicles along the road,
       then compute each one's best-connected BS.
    2) update_positions(): move forward by fixed speed each round,
       then recompute connectivity.
    """

    def __init__(self):
        self.positions = {}    # e.g. {"VE_0": 0.0, "VE_1": 200.0, ...}
        self.connected_bs = {} # e.g. {"VE_0": (0, -45.3), ...}

    def initialize_positions(self):
        """
        Initialize vehicle positions based on the maximum count from workload range.
        """
        # Use the upper bound of num_vehicles_range as the total vehicle count
        max_veh = cfg.WORKLOAD['num_vehicles_range'][1]
        # Evenly space vehicles along the road length
        spacing = cfg.ROAD_LENGTH / max_veh
        for i in range(max_veh):
            vid = f"VE_{i}"
            pos = i * spacing
            offset = random.uniform(-0.1, 0.1) * spacing
            self.positions[vid] = (pos + offset) % cfg.ROAD_LENGTH
            self.update_connectivity(vid)  # ✅ still required


    def update_positions(self):
        """
        Move each vehicle forward by a fixed speed and update connectivity.
        """
        max_veh = cfg.WORKLOAD['num_vehicles_range'][1]
        for i in range(max_veh):
            vid = f"VE_{i}"
            # Retrieve current position (default to 0 if missing)
            pos = self.positions.get(vid, 0.0)
            # Move forward by mean speed and wrap around road length
            speed = cfg.MOBILITY_CONFIG['vehicle_speed_mean']
            new_pos = (pos + speed) % cfg.ROAD_LENGTH
            self.positions[vid] = new_pos
            self.update_connectivity(vid)

    def update_connectivity(self, vid):
        """
        Determine the best-connected base station for the given vehicle.
        """
        best_idx, best_sig = None, float('-inf')
        for idx, bspos in enumerate(cfg.BS_POSITIONS):
            dist = self.circular_distance(self.positions[vid], bspos)
            #dist = abs(self.positions[vid] - bspos)
            sig = calculate_signal_strength(dist)
            if sig > best_sig:
                best_sig, best_idx = sig, idx
        self.connected_bs[vid] = (best_idx, best_sig)

    def circular_distance(self, pos1, pos2):
        diff = abs(pos1 - pos2)
        return min(diff, cfg.ROAD_LENGTH - diff)
