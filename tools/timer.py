import logging
import time
import threading
from functools import wraps, partial
import psutil
import os
import sys

# Cache the process object to avoid recreating it on every call
_process = psutil.Process(os.getpid())

def get_memory_usage():
    """
    Returns memory usage in MB.
    Attempts to include swap usage where possible without significant performance penalty.
    """
    mem = _process.memory_info()
    
    # Windows: 'private' field in memory_info includes swap (commit charge) and is fast
    if hasattr(mem, 'private'):
        return mem.private / (1024 * 1024)
        
    # Linux: Read VmSwap from /proc to avoid expensive memory_full_info()
    if sys.platform.startswith('linux'):
        try:
            with open(f'/proc/{_process.pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('VmSwap:'):
                        # Format: VmSwap:        1234 kB
                        swap_kb = int(line.split()[1])
                        return (mem.rss + swap_kb * 1024) / (1024 * 1024)
        except (IOError, ValueError, IndexError):
            pass
            
    # Fallback (macOS, etc.): Return RSS only
    # memory_full_info() which provides swap is too slow on macOS (calculates USS)
    return mem.rss / (1024 * 1024)

class TimerContext:
    def __init__(self):
        self.start_time = 0
        self.accumulated_time = 0
        self.is_paused = False
        self.last_pause_time = 0

    def start(self):
        self.start_time = time.time()
        self.is_paused = False

    def pause(self):
        if not self.is_paused:
            self.accumulated_time += time.time() - self.start_time
            self.is_paused = True

    def resume(self):
        if self.is_paused:
            self.start_time = time.time()
            self.is_paused = False

    def get_duration(self):
        if not self.is_paused:
            return self.accumulated_time + (time.time() - self.start_time)
        return self.accumulated_time

def timer(func=None, *, log=True, threaded=True, independent=False, memory=True):
    if func is None:
        return partial(timer, log=log, threaded=threaded, independent=independent, memory=memory)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        method_name = func.__name__

        identifier = f"{class_name}.{method_name}"
        
        start_mem = 0
        if memory:
            start_mem = get_memory_usage()
            
        if log:
            mem_msg = f" (Memory: {start_mem:.2f} MB)" if memory else ""
            logging.info(f"Starting execution of {identifier}...{mem_msg}")

        peak_memory = [start_mem]
        stop_event = threading.Event()

        def monitor():
            while not stop_event.is_set():
                current_mem = get_memory_usage()
                if current_mem > peak_memory[0]:
                    peak_memory[0] = current_mem
                time.sleep(0.1)

        t = None
        if threaded and memory:
            t = threading.Thread(target=monitor)
            t.start()

        current_timer = TimerContext()

        if not independent:
            # Initialize timer stack if not present
            if not hasattr(self, '_timer_stack'):
                self._timer_stack = []

            # Pause parent timer if exists
            if self._timer_stack:
                self._timer_stack[-1].pause()

            self._timer_stack.append(current_timer)
        
        current_timer.start()

        try:
            result = func(self, *args, **kwargs)
        finally:
            if t:
                stop_event.set()
                t.join()
            
            # Stop current timer
            duration = current_timer.get_duration()
            
            if not independent:
                self._timer_stack.pop()
                
                # Resume parent timer if exists
                if self._timer_stack:
                    self._timer_stack[-1].resume()

        if memory:
            end_mem = get_memory_usage()
            if end_mem > peak_memory[0]:
                peak_memory[0] = end_mem

        if log:
            mem_msg = f" (Peak Memory: {peak_memory[0]:.2f} MB)" if memory else ""
            logging.info(f"Finished execution of {identifier} in {duration:.2f} seconds.{mem_msg}")

        # Accumulate duration for repeated calls
        self.builder.benchmarking.add_row(self.builder.run_id, identifier, duration, accumulate=True)

        if memory:
            # Tell the benchmark class to strictly use the maximum peak observed
            self.builder.memory_benchmarking.add_row(
                self.builder.run_id,
                f"{identifier}_peak_memory_mb",
                peak_memory[0],
                mode='max'
            )

        return result
    return wrapper