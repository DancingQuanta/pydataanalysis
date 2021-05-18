import os
import numpy as np

def mkdir(*args):
    dir = os.path.join(*args)
    os.makedirs(dir, exist_ok=True)
    return dir

def is_odd(num):
    return num & 0x1

def odict_sort_by_list(d, sort_list):
    return type(d)((k, d[k]) for k in sort_list)

def parse_label(s):
    lst = (s.split(' ('))
    if len(lst) == 1:
        return lst[0], ""
    elif len(lst) == 2:
        return lst[0], lst[1].rstrip(')')


def csnap(df, fn=lambda x: x, msg=None):
    """ Custom Help function to print things in method chaining.
        Returns back the df to further use in chaining.
    """

    if msg:
        print(msg)
    display(fn(df))

    return df


def vrange(starts, stops):
    """Create concatenated ranges of integers for multiple start/stop
    https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = [1, 3, 4, 6]
        >>> stops  = [1, 5, 7, 6]
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    stops = np.asarray(stops)
    l = stops - starts # Lengths of each range.
#     print(l.sum())
    return np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())

def percent_change(data, old, new):
    """
    Calculate a percentage change from old to new
    Args:
        data: dictionary
        old: string of a dictionary key 
        new: string of a dictionary key
    """
    # Accept a dictionary
    percent = ((data[new] - data[old])
               /data[old]) * 100
    return percent

def percent_diff(data, one, two):
    """
    Calculate a difference
    Args:
        data: dictionary
        one: string of a dictionary key 
        two: string of a dictionary key
    """
    avg = (data[one] + data[two])/2
    diff = np.abs(data[one] - data[two])
    percent = (diff / avg) * 100
    
    return percent

def percent_error(data, exact, approx):
    """
    Calculate a percentage error of approx with exact
    Args:
        data: dictionary
        exact: string of a dictionary key 
        approx: string of a dictionary key
    """
    error = np.abs(data[exact] - data[approx])
    percent = (error / data[exact]) * 100
    
    return percent

def sqrt_ss(arr):
    """Square root of sum of squares"""
    return np.sqrt(np.sum(np.square(arr)))
