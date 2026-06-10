import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import sys
from time import time as clock
from tqdm import tqdm

def _Get_Colours(num):
    """
    Used for colouring sightlines blue -> red.
    """

    if num < 3 :
        colours = ['blue','red']
    elif num == 3:
        colours = ['blue','forestgreen','red']
    elif num == 4:
        colours = ['blue','forestgreen','tab:orange','red']
    else:
        cmap = plt.cm.jet  
        cmaplist = np.array([cmap(i) for i in range(cmap.N)])
        colours = cmaplist[np.linspace(30,len(cmaplist)-30,num).astype(int)]
    return colours


@njit
def _Counting_Sort(flat, n_voxels):
    counts = np.zeros(n_voxels, dtype=np.int64)
    for k in flat:
        if k >= 0:
            counts[k] += 1

    offsets = np.zeros(n_voxels + 1, dtype=np.int64)
    for i in range(n_voxels):
        offsets[i + 1] = offsets[i] + counts[i]

    order = np.empty(offsets[-1], dtype=np.int64)
    pos   = offsets[:-1].copy()
    for i in range(len(flat)):
        k = flat[i]
        if k >= 0:
            order[pos[k]] = i
            pos[k] += 1

    return order, offsets

def _Progress_Print(msg,time_start):

    if sys.stdout.isatty():
        print(f"{msg} -- Done ({clock()-time_start:.0f}s)")
    else:
        print(f" -- Done ({clock()-time_start:.0f}s)")

def _Smart_Tqdm(iterable, desc="", total=None, every_sec=60):
    
    if sys.stdout.isatty():
        yield from tqdm(iterable, desc=desc, total=total)
        return
    
    if total is None:
        iterable = list(iterable)
        total = len(iterable)
    
    t_start = clock()
    t_last = t_start

    for i, item in enumerate(iterable):
        yield item
        t_now = clock()
        if t_now - t_last >= every_sec:
            pct = int((i + 1) / total * 100)
            elapsed = t_now - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            rate_str = f"{rate:.1f} it/s" if rate < 1000 else f"{rate/1000:.2f}k it/s"
            print(f"[{pct:3d}%] {desc} ({i+1}/{total}, {rate_str})", file=sys.stderr)
            t_last = t_now
    
    elapsed = clock() - t_start
    print(f"[100%] {desc} ({total}/{total}, {elapsed:.1f}s total)", file=sys.stderr)