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
import pandas as pd
import xarray as xr

import holoviews as hv
from holoviews import opts, dim
import hvplot.pandas
import hvplot.xarray

hv.extension('bokeh')
# hv.notebook_extension(display_formats=['html', 'svg'])
# hv.output(fig='svg')

# Dimensions
I = xr.DataArray(np.arange(10), dims='I', name='I')
direction = xr.DataArray(['negative', 'positive'], dims='direction', name='direction')
reps = xr.DataArray(np.arange(5), dims='reps', name='reps')
idx = np.arange(2000)

x = xr.DataArray(np.random.randn(len(I), len(direction), len(reps), len(idx)),
                 dims=['I', 'direction', 'reps', 'idx'],
                 name='x')

groupby = ['I', 'direction', 'reps']
kind = 'scatter'
plot = x.hvplot(x='idx', y='x', groupby=groupby, kind=kind, dynamic=False)
hv.render(plot)
```

```python
y = (data.to_dataframe().reset_index()
     .hvplot.scatter(x='index', y='y', groupby=['I', 'direction', 'reps'], dynamic=False))
plot = x + y
plot = plot.opts(opts.Scatter(width=300))
```

```python
%%prun
hv.render(plot)
```

```python
plot = (data.to_dataframe()
 .hvplot.scatter(x='x', y='y', groupby=['I', 'direction', 'reps'], dynamic=False))
plot.opts(width=300)
```
