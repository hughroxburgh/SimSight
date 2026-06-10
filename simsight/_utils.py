import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import sys
from time import time as clock

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