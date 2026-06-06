import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, config as C

if __name__ == "__main__":
    print(f"[build] N_TRAIN={C.N_PER_CLASS_TRAIN} N_VAL={C.N_PER_CLASS_VAL} "
          f"N_TEST={C.N_PER_CLASS_TEST} (por classe, {C.NUM_CLASSES} classes)")
    t0 = time.time()
    names = data.build_all_subsets()
    print(f"[build] concluído em {time.time()-t0:.0f}s")
    print(f"[build] classes ({len(names)}): {names}")
