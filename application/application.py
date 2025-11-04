import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.test2 import AIScheduler
from core.simulator import Simulator

# CHOOSE MODE HERE
TRAIN = False  # Set to False for evaluation mode                    

# Instantiate scheduler and simulator
scheduler = AIScheduler()
scheduler.set_train_mode(TRAIN)   # ← REQUIRED!
sim = Simulator(scheduler)

# Run simulation
sim.run(train_mode=TRAIN)
