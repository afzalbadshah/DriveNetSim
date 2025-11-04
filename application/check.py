import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scheduler.test2 import AIScheduler
from core.simulator import Simulator

# CHOOSE MODE HERE
TRAIN = True   # ← set False to just run one evaluation

# Instantiate scheduler
scheduler = AIScheduler()

if TRAIN:
    scheduler.set_train_mode(True)

    # Loop epochs until zero wrong decisions (or cap)
    scheduler.train_until_perfect(
        simulator_class=Simulator,
        max_epochs=2000,    # safety cap
        tolerance=10,       # zero wrong decisions
        streak=3           # require 3 perfect epochs in a row
    )
else:
    scheduler.set_train_mode(False)
    sim = Simulator(scheduler)
    sim.run(train_mode=False)
