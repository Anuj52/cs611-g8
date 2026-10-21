"""Load W_C and X_cal.

Looks for W_C.npy and X_cal.npy in ./data/ or next to the scripts.
Override with: python3 explore.py --data /path/to/folder
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    sys.exit(f"Could not find {name}. Put it in {os.path.join(HERE, 'data')} or pass --data <folder>.")


def load():
    dirs = [os.path.join(HERE, "data"), HERE]
    if "--data" in sys.argv:
        dirs.insert(0, sys.argv[sys.argv.index("--data") + 1])
    W = np.load(_find("W_C.npy", dirs)).astype(np.float64)
    X = np.load(_find("X_cal.npy", dirs)).astype(np.float64)
    return W, X
