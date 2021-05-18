import numpy as np

def magnitude_thousand(value):
    """
    Find the exponent of the input value and then nearest thousands.

    Args:
        value (float): input value

    Returns:
        float: exponent
    """
    # Taking the exponent
    exponent = int(np.log10(np.abs(value)))

    # If the exponent is smaler than 0 dividing the exponent with -1

    if exponent < 0:
        exponent = exponent-1

    # Returns magnitude in steps of 3
    exponent = (exponent // 3) * 3

    return exponent

def scientific_notation(arr, units='', template='1e{exponent}'):
    """
    Scales the data to thousands and append the original exponent to the units

    Args:
        arr (array-like): Array of data

    Keyword Args:
        units (string): units of the data which the axponent will be append to.

    Returns:
        array-like: Scaled data
        units: Modified units with exponent
    """

    # Get max
    value = arr.max()

    # Calculate thousand muliplter
    exponent = magnitude_thousand(value)

    # Adjust scale

    if exponent != 0:

        # Divide array by exponent.
        arr = arr / 10**exponent

        # Convert exponent into a formated string with a template
        exponent = template.format(exponent=exponent)

        # Add exponent to units string

        if units:
            units = units + ' ' + exponent
        else:
            units = exponent

    return arr, units
