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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import holoviews as hv
from holoviews import opts, dim
import hvplot.pandas
import hvplot.xarray

hv.extension('bokeh')
I = xr.DataArray(np.arange(10), dims='I', name='I')
count = xr.DataArray(np.arange(5), dims='count', name='count')
direction = xr.DataArray(['negative', 'positive'], dims='direction', name='direction')
idx = xr.DataArray(np.arange(100), dims='idx', name='idx')
line = xr.DataArray(['Left', 'Right'], dims='line', name='line')

# mu = xr.DataArray(np.random.randn(len(I), len(count)),
#                   coords=[I, count],
#                   name='mu')
field = xr.DataArray(np.random.randn(len(I), len(count), len(direction), len(idx)),
                     coords=[I, count, direction, idx],
                     name='field')

load = xr.DataArray(np.random.randn(len(I), len(count), len(direction), len(idx)),
                    coords=[I, count, direction, idx],
                    name='load')

data = xr.merge([field, load])

field_line = xr.DataArray(np.random.randn(len(I), len(count), len(direction), len(line)),
                          coords=[I, count, direction, line],
                          name='field_line')

load_line = xr.DataArray(np.random.randn(len(I), len(count), len(direction), len(line)),
                          coords=[I, count, direction, line],
                          name='load_line')

line = xr.merge([field_line, load_line])
line

def compute_marker(scatter):
    direction_markers = {
        'positive': 'circle',
        'negative': 'triangle'
    }

    sel_data = scatter.data
    direction = sel_data['direction'].iloc[0]
    return scatter.opts(size=3, marker=direction_markers[direction])

scatter = (data.to_dataframe()
           .hvplot.scatter(x='field', y='load', groupby=['I', 'count', 'direction'], dynamic=False)
           .apply(compute_marker))

curve = (line.to_dataframe()
         .hvplot.line(x='field_line', y='load_line', groupby=['I', 'count', 'direction'], dynamic=False))

plot = scatter * curve
plot
```
