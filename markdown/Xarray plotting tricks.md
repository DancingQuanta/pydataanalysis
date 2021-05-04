```python
import numpy as np
import xarray as xr
from scipy.stats import norm
from matplotlib import pyplot as plt

# nominal x
x = xr.DataArray(np.arange(-10, 10, 0.1), dims='index', name='x')

# Shift loc
loc = xr.DataArray(np.arange(-0.5, 1, 0.5), dims='loc', name='loc')

# Number of experiments
exp = xr.DataArray(range(3), dims='exp', name='exp')

# Add noise to x per experiment
noise = xr.DataArray(np.random.rand(len(x), len(loc)),
                     coords={'loc': loc},
                     dims=['index', 'loc'])
x = x + noise * 0.5

# Measure
y = xr.apply_ufunc(
        norm.pdf, x, x['loc'], 1,
        input_core_dims=[['index'], [], []],
        output_core_dims=[['index']],
        vectorize=True
    )

# Name
x.name = 'x'
y.name = 'y'
```


## `Dataarray.plot.line`

Plot DataArray against `index` for every `loc` as line plot

    y.plot.line(x='index', hue='loc')

```python
# Plot only y
y.plot.line(x='index', hue='loc')
```

## `Dataset.plot.scatter`

Plot `y` against `x` in Dataset for every `loc` as line plot

    data.plot.scatter(x='x', y='y', hue='loc', hue_style='discrete')

```python
# Plot y against x
data = xr.merge([x, y])
data.plot.scatter(x='x', y='y', hue='loc', hue_style='discrete')
```
