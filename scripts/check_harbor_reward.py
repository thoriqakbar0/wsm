from pathlib import Path
import sys

rewards = list(Path(sys.argv[1]).rglob("reward.txt"))
if len(rewards) != 1 or float(rewards[0].read_text()) != 1.0:
    raise SystemExit("Expected exactly one Harbor oracle reward of 1.0")
print("Native Harbor oracle reward is 1.0")
