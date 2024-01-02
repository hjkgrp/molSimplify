import time


class DebugTimer:
    """Context used to time the call to various ML models and print the measurement"""
    def __init__(self, name: str, print: bool = True):
        self.name = name
        self.print = print

    def __enter__(self):
        self.t_start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.t_stop = time.perf_counter()
        if self.print:
            print(f'{self.name} took {self.t_stop - self.t_start:.2f} seconds')
