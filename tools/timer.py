import logging
import time
import threading
from functools import wraps, partial
import psutil
import os

# Cache the process object to avoid recreating it on every call
_process = psutil.Process(os.getpid())

def get_memory_usage():
    return _process.memory_info().rss / (1024 * 1024)

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

def timer(func=None, *, log=True, threaded=True, independent=False):
    if func is None:
        return partial(timer, log=log, threaded=threaded, independent=independent)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        method_name = func.__name__
        
        if 'ConceptClusterer' in class_name:
            identifier = method_name
        else:
            identifier = f"{class_name}.{method_name}"
        
        start_mem = get_memory_usage()
        if log:
            logging.info(f"Starting execution of {identifier}... (Memory: {start_mem:.2f} MB)")

        peak_memory = [start_mem]
        stop_event = threading.Event()

        def monitor():
            while not stop_event.is_set():
                current_mem = get_memory_usage()
                if current_mem > peak_memory[0]:
                    peak_memory[0] = current_mem
                time.sleep(0.1)

        t = None
        if threaded:
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

        end_mem = get_memory_usage()
        if end_mem > peak_memory[0]:
            peak_memory[0] = end_mem

        if log:
            logging.info(f"Finished execution of {identifier} in {duration:.2f} seconds. (Peak Memory: {peak_memory[0]:.2f} MB)")

        self.builder.benchmarking.add_row(self.builder.run_id, identifier, duration)
        self.builder.memory_benchmarking.add_row(self.builder.run_id, f"{identifier}_peak_memory_mb", peak_memory[0])
        
        return result
    return wrapper
