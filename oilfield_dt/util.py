import time
import numpy as np


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ends_by_ep(ds, n):
    return [ds["ends"][ds["eid"] == k] for k in range(n)]


def meta_of(eps):
    return [dict(label=e["label"], mode=e["mode"], onset=e["onset"]) for e in eps]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
