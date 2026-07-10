import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import sys
from time import time as clock
from tqdm import tqdm
import gc
import ctypes

def _Get_Colours(num,dark_mode=False):
    """
    Used for colouring sightlines blue -> red.
    """

    if dark_mode:
        if num < 3:
            colours = ['deepskyblue', 'tomato']
        elif num == 3:
            colours = ['deepskyblue', 'mediumspringgreen', 'tomato']
        elif num == 4:
            colours = ['deepskyblue', 'mediumspringgreen', 'gold', 'tomato']
    else:
        if num < 3:
            colours = ['dodgerblue', 'indianred']
        elif num == 3:
            colours = ['dodgerblue', 'mediumseagreen', 'indianred']
        elif num == 4:
            colours = ['dodgerblue', 'mediumseagreen', 'tab:orange', 'indianred']

    if num >= 5:
        cmap = plt.cm.jet if dark_mode else plt.cm.turbo
        cmaplist = np.array([cmap(i) for i in range(cmap.N)])
        colours = cmaplist[np.linspace(30, len(cmaplist) - 30, num).astype(int)]

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

def _Is_Interactive():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except NameError:
        return sys.stdout.isatty()

def _Progress_Print(msg, time_start):

    def _mem_str():
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / 1e9
            return f", mem={rss_gb:.2f}GB"
        except ImportError:
            try:
                import resource
                peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_gb = peak_kb / 1e6 if sys.platform != 'darwin' else peak_kb / 1e9
                return f", peak_mem={peak_gb:.2f}GB"
            except Exception:
                return ""

    mem_str = _mem_str()

    if _Is_Interactive():
        print(f"{msg} -- Done ({clock()-time_start:.0f}s{mem_str})", flush=True)
    else:
        print(f" -- Done ({clock()-time_start:.0f}s{mem_str})", flush=True)

def _Smart_Tqdm(iterable, desc="", total=None, every_sec=60):

    def _mem_str():
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / 1e9
            return f", mem={rss_gb:.2f}GB"
        except ImportError:
            try:
                import resource
                # ru_maxrss is KB on Linux, bytes on macOS
                peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_gb = peak_kb / 1e6 if sys.platform != 'darwin' else peak_kb / 1e9
                return f", peak_mem={peak_gb:.2f}GB"
            except Exception:
                return ""

    if _Is_Interactive():
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
            mem_str = _mem_str()
            print(f"[{pct:3d}%] {desc} ({i+1}/{total}, {rate_str}{mem_str})", file=sys.stderr)
            t_last = t_now

    elapsed = clock() - t_start
    mem_str = _mem_str()
    print(f"[100%] {desc} ({total}/{total}, {elapsed:.1f}s total{mem_str})", file=sys.stderr)
    print('\n', file=sys.stderr)



def Cleanup_Memory(verbose=False):
    """
    Force garbage collection and return freed memory to the OS where possible.
    Safe to call periodically in long-running jobs (e.g. between sightlines/snapshots)
    to prevent glibc's malloc arenas from accumulating freed-but-unreturned memory.
    """
    if verbose:
        import psutil, os
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss / 1e9

    # clear any dangling exception state that can pin whole call stacks
    sys.last_traceback = None
    sys.last_value = None
    sys.last_type = None

    collected = gc.collect()

    trimmed = 0
    if sys.platform.startswith('linux'):
        try:
            libc = ctypes.CDLL("libc.so.6")
            trimmed = libc.malloc_trim(0)
        except OSError:
            pass  # not available on this platform/libc

    if verbose:
        rss_after = proc.memory_info().rss / 1e9
        print(f"    [cleanup_memory] gc collected {collected} objects, "
              f"malloc_trim={'freed' if trimmed else 'nothing to free'}, "
              f"RSS {rss_before:.2f} GB -> {rss_after:.2f} GB", flush=True)

    return trimmed