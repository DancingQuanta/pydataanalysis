---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Scientific notation

It can be beneficial to display data with a limited number of significant figures in a scientific notation to emphasis the significant figures and their exponent and for readability.

iN this section, data in various data structures are manipulated and plotted to show the scientific notation.

```python
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
```

Using numpy and matplotlib.

Generate large valued data and plot it

```python
# Generate large valued data
x = np.arange(10) * 1e6
y = x ** 2
```

```python
plt.plot(x, y)
```

It can be seen that matplotlib already scales the data and attach their scientific notation to the corners of the plot.

What if we wanted them in the ylabel? First use the funciton `scientific_notation` to scale the data to thousandas and return the scientific notation.

```python
from pydataanalysis.numpy_utils import scientific_notation

x, xunits = scientific_notation(x)
y, yunits = scientific_notation(y)
```

```python
plt.plot(x, y)
plt.xlabel(xunits)
plt.ylabel(yunits)
```

Using pandas

```python
import pandas as pd

df = pd.DataFrame({'x': x, 'y': y})

df.plot(x='x', y='y')
plt.xlabel(xunits)
plt.ylabel(yunits)
```

Using xarray

```python
import xarray as xr

# Generate large valued data
x = np.arange(10) * 1e6
y = x ** 2

# Create DataArray object
da = xr.DataArray(y,
                  coords={'x': x},
                  dims=['x'],
                  name='y')

# Add units
da.x.attrs['units'] = 'V'
da.attrs['units'] = 'I'

# Plot original data
da.plot(x='x')
```

```python
from pydataanalysis.xarray_utils import xscientific_notation

# Scale data
da = da.pipe(xscientific_notation)
da['x'] = da['x'].pipe(xscientific_notation)

# Plot
da.plot(x='x')
```
