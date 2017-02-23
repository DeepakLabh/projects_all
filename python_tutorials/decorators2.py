#!/usr/bin/env python
def benchmark(func):
    """
    Print the seconds that a function takes to execute.
    """
    from time import time
    def wrapper(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        print("function @{0} took {1:0.3f} seconds".format(func.__name__, time() - t0))
        return res
    return wrapper
    
@benchmark
def wait_some_seconds(num_seconds = 1):
    from time import sleep
    sleep(num_seconds)

wait_some_seconds(1.11)
# function @wait_some_seconds took 1.11 seconds
wait_some_seconds(5)
# function @wait_some_seconds took 5.000 seconds
