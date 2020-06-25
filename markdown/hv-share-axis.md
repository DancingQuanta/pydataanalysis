---
jupyter:
  jupytext:
    formats: markdown//md,notebooks//ipynb,scripts//py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import xarray as xr
```

```python
from src.plot import *
import matplotlib.pyplot as plt
# import holoviews as hv
# from holoviews import opts
# import hvplot.xarray

# hv.extension('matplotlib')
# hv.notebook_extension(display_formats=['html', 'svg'])
# hv.output(fig='svg')

# Dimensions
N = xr.DataArray(np.arange(100), dims='N', name='N')
reps = xr.DataArray(np.arange(5), dims='reps', name='reps')
horizon = xr.DataArray([1, -1], dims='horizon', name='horizon')
horizon.attrs = {'long_name': 'Horizonal', 'units': 'H'}
vertical = xr.DataArray(np.arange(1, 4), dims='vertical', name='vertical')
vertical.attrs = {'long_name': 'Vertical', 'units': 'V'}

x = xr.DataArray(np.random.randn(len(N), len(reps)),
                 dims=['N', 'reps'],
                 name='x')
y = vertical * horizon * x
y.name = 'y'

# Merge x, y
data = xr.merge([x, y])

# Assign coords
data = data.assign_coords(reps=reps, vertical=vertical, horizon=horizon)

# grid = data.hvplot.scatter(x='x', y='y', row='vertical', col='horizon', groupby=[])
# grid.opts(initial_hooks=[test_hook])
```

```python

x = 'Bext'
y = 'y1'
dim = 'N'
col = 'direction'
row = 'I'

data['x'] = data['x'].broadcast_like(data['y'])

def axes_lims(ds, x, y, row, col):
    row_dims = list(ds.dims)
    col_dims = row_dims.copy()
    row_dims.remove(col)
    col_dims.remove(row)

#     print(ds.dims)
#     print(row_dims)
    x_min = ds[x].min(dim=row_dims, skipna=True)
    x_max = ds[x].max(dim=row_dims, skipna=True)

    y_min = ds[y].min(dim=col_dims, skipna=True)
    y_max = ds[y].max(dim=col_dims, skipna=True)

    x = xr.concat([x_min, x_max], xr.IndexVariable('lim', ['min', 'max']))
    y = xr.concat([y_min, y_max], xr.IndexVariable('lim', ['min', 'max']))
    ds = xr.Dataset({'x': x, 'y': y})
    return ds
axes_lims(data, 'x', 'y', 'horizon', 'vertical')


fig = mpl_grid(data, 'x', 'y', 'reps', 'horizon', 'vertical')
# row_dims
# # [[x_max, x_min], [y_max, y_min]]

# def mpl_grid(ds, row, col, func):

#     # Number of rows and cols
#     nrows = len(ds[row])
#     ncols = len(ds[col])

#     fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4), sharey=True, sharex=True)

#     print(axes)
```

```python
# data.hvplot.scatter(x='x', y='y', row='vertical', col='horizon', groupby=[]).opts(axiswise=True)#.opts(opts.Scatter(axiswise=True))
# hv.Dataset(data)#.to.scatter('x', 'y', groupby=['vertical', 'horizon'])#.grid('horizon', 'vertical')
```

```python
# hv.help(hv.GridSpace)
```
