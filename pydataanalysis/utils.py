import os
import numpy as np

def mkdir(*args):
    dir = os.path.join(*args)
    os.makedirs(dir, exist_ok=True)
    return dir

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

def magnitude_thousand(number):
    # Checking if its negative or positive
    if number < 0:
        negative = True
    else:
        negative = False

    # if its negative converting to positive (math.log()....)
    if negative:
        number = number * -1

    # Taking the exponent
    exponent = int(np.log10(number))

    # Checking if it was negative converting it back to negative
    if negative:
        number = number * -1

    # If the exponent is smaler than 0 dividing the exponent with -1
    if exponent < 0:
        exponent = exponent-1
    
    # Returns magnitude in steps of 3
    exponent = (exponent // 3) * 3
    return exponent

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
