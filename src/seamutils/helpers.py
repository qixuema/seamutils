import datetime
import functools
import sys
import time
from pathlib import Path
from collections import defaultdict
import string
import random

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def first(it):
    return it[0]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def exists(x):
    return x is not None

def get_current_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted_time

def cycle(dl):
    while True:
        for data in dl:
            yield data

def divisible_by(num, den):
    return num % den == 0

def is_odd(n):
    return not divisible_by(n, 2)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def is_debug():
    return True if sys.gettrace() else False

def get_file_list_with_extension(folder_path, ext):
    """
    Search for all files with the specified extension(s) in the given folder and its subfolders.

    Args:
    folder_path (str): Path to the folder where the search will be performed.
    ext (str or list of str): File extension(s) to search for, starting with a dot (e.g., '.ply').

    Returns:
    list: A list of file paths (in POSIX format) matching the specified extension(s).
    """
    file_path_list_with_extension = []

    # Ensure 'ext' is a list
    if isinstance(ext, str):
        ext = [ext]

    ext_set = {e.lower() for e in ext}

    folder_path = Path(folder_path)

    # Traverse all files recursively
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ext_set:
            file_path_list_with_extension.append(file_path.as_posix())
    
    return file_path_list_with_extension


class Timer(object):

    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def tictoc(self, diff):
        self.diff = diff
        self.total_time += diff
        self.calls += 1

    def total(self):
        """ return the total amount of time """
        return self.total_time

    def avg(self):
        """ return the average amount of time """
        return self.total_time / float(self.calls)

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

class Timers(object):

    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def tictoc(self, key, diff):
        self.timers[key].tictoc( diff)

    def print(self, key=None):
        if key is None:
            # print all time
            for k, v in self.timers.items():
                print("{:}: \t  average {:.4f},  total {:.4f} ,\t calls {:}".format(k.ljust(30),  v.avg(), v.total_time, v.calls))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))

    def get_print_string(self):
        strings = []
        for k, v in self.timers.items():
            strings.append("{:}: \t  average {:.4f},  total {:.4f} ,\t calls {:}".format(k.ljust(30),  v.avg(), v.total_time, v.calls))        
        string = "\n".join(strings)
        return string

    def get_avg(self, key):
        return self.timers[key].avg()
    
    
def generate_random_string(length, batch_size=1):
    chars = string.ascii_letters + string.digits
    pool = random.choices(chars, k=length * batch_size)

    return [''.join(pool[i*length:(i+1)*length]) for i in range(batch_size)]