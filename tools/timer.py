import logging
import time
from functools import wraps, partial
import tracemalloc


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

        start_mem_mb = 0
        if memory:
            # Start tracing if it hasn't been started globally yet
            if not tracemalloc.is_tracing():
                tracemalloc.start()

            # Reset the peak so we measure strictly from the start of this function
            tracemalloc.reset_peak()
            current_mem, _ = tracemalloc.get_traced_memory()
            start_mem_mb = current_mem / (1024 * 1024)

        if log:
            mem_msg = f" (Memory: {start_mem_mb:.2f} MB)" if memory else ""
            logging.info(f"Starting execution of {identifier}...{mem_msg}")

        current_timer = TimerContext()

        if not independent:
            if not hasattr(self, '_timer_stack'):
                self._timer_stack = []

            if self._timer_stack:
                self._timer_stack[-1].pause()

            self._timer_stack.append(current_timer)

        current_timer.start()

        try:
            result = func(self, *args, **kwargs)
        finally:
            duration = current_timer.get_duration()

            if not independent:
                self._timer_stack.pop()
                if self._timer_stack:
                    self._timer_stack[-1].resume()

            # Capture peak memory immediately in the finally block
            peak_memory_mb = 0
            if memory:
                _, peak_mem = tracemalloc.get_traced_memory()
                peak_memory_mb = peak_mem / (1024 * 1024)

        if log:
            mem_msg = f" (Peak Python Memory: {peak_memory_mb:.2f} MB)" if memory else ""
            logging.info(f"Finished execution of {identifier} in {duration:.2f} seconds.{mem_msg}")

        self.builder.benchmarking.add_row(self.builder.run_id, identifier, duration, accumulate=True)

        if memory:
            self.builder.memory_benchmarking.add_row(
                self.builder.run_id,
                f"{identifier}_peak_memory_mb",
                peak_memory_mb,
                mode='max'
            )

        return result

    return wrapper