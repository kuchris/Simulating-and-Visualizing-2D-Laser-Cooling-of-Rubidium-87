"""Run independent simulations side by side, one per CPU core.

Parameter scans are many separate simulations. `parallel_map` hands them to a
pool of worker processes; each worker runs the compiled loop of `fast` on a
single thread, so the cores are not oversubscribed. The results are the same
as running them one after another.

Set the environment variable MOT_PROCESSES=1 to run everything in this process.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from . import fast

SINGLE_THREAD = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                        "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")}


def available_processes() -> int:
    return int(os.environ.get("MOT_PROCESSES", os.cpu_count() or 1))


def _start_worker():
    fast.set_threads(1)


def parallel_map(func, items, processes: int | None = None) -> list:
    """[func(item) for item in items], spread over worker processes.

    `func` must be importable by the workers: a module-level function, or a
    functools.partial of one.
    """
    items = list(items)
    n = min(processes or available_processes(), len(items))
    if n <= 1:
        return [func(item) for item in items]
    fast.warm_up()  # compile once here, so the workers load the cached machine code
    context = multiprocessing.get_context("spawn")
    # Workers inherit the environment when they start: one thread each for the
    # numerical libraries, or n workers would each start a thread per core.
    saved = {name: os.environ.get(name) for name in SINGLE_THREAD}
    os.environ.update(SINGLE_THREAD)
    try:
        with ProcessPoolExecutor(n, mp_context=context, initializer=_start_worker) as pool:
            return list(pool.map(func, items))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
