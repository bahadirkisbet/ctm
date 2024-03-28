import glob
import os

import numpy as np
from os import listdir
from os.path import isdir, join


def list_files(directory):
    from os import listdir
    from os.path import isfile, join
    return [f"{directory}/{list_result}" for list_result in listdir(directory) if
            not (not isfile(join(directory, list_result)) or not list_result.endswith(".mp4"))]


def get_all_files(path, extension=".mp4"):
    return [os.path.join(r, fn)
            for r, ds, fs in os.walk(path)
            for fn in fs if fn.endswith(extension)]
def list_directories(directory):
    return [f"{directory}/{list_result}" for list_result in listdir(directory) if
            not (not isdir(join(directory, list_result)))]


def find_local_minima_after_max(arr):
    # Step 1: Find the index of the maximum value
    max_index = arr.index(max(arr))

    # Step 2: Iterate through elements after the maximum value and find local minima
    local_minima = []
    for i in range(max_index + 1, len(arr) - 1):
        if arr[i - 1] > arr[i] < arr[i + 1]:
            local_minima.append((i, arr[i]))

    return local_minima


def find_local_minima_before_max(arr):
    # Step 1: Find the index of the maximum value
    max_index = arr.index(max(arr))

    # Step 2: Iterate through elements after the maximum value and find local minima
    local_minima = []
    for i in range(1, max_index):
        if arr[i - 1] > arr[i] < arr[i + 1]:
            local_minima.append((i, arr[i]))

    return local_minima


def trim_left(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            return arr[i:]
    return arr


def get_first_non_zero_index(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            return i
    return 0


def trim_right(arr):
    for i in range(len(arr)):
        if arr[i] == 0:
            return arr[:i]
    return arr


def calculate_velocity(arr_x, arr_y):
    velocity = []
    for i in range(len(arr_x)):
        if i != 0:
            velocity.append(np.sqrt(pow(arr_x[i] - arr_x[i - 1], 2) + pow(arr_y[i] - arr_y[i - 1], 2)))
    return velocity


def get_moving_average(arr: list, index: int, window_size: int = 3):
    window = arr[index - window_size:index]
    if len(window) < window_size:
        return 0
    return sum(window) / window_size


def get_diff(arr):
    diff = []
    for i in range(len(arr)):
        if i != 0:
            diff.append(arr[i] - arr[i - 1])
    return diff


def get_local_minima_after_max(arr):
    peak_index = arr.index(max(arr))

    for i in range(peak_index + 1, len(arr)):
        if arr[i] < arr[i - 1]:
            return i, arr[i]
