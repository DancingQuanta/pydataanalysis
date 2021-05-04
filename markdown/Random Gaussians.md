---
jupyter:
  jupytext:
    formats: markdown//md,notebooks//ipynb,scripts//py:percent
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

```python
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats

import holoviews as hv
from holoviews import opts, dim
import hvplot.pandas
import hvplot.xarray

hv.extension('bokeh')
# hv.notebook_extension(display_formats=['html', 'svg'])
# hv.output(fig='svg')

# from src.import_data import *
```

```python
hv.__version__
```

```python
x = xr.DataArray(np.arange(5), dims='x', name='x')
y = xr.DataArray(np.arange(5), dims='y', name='y')
# mu = xr.DataArray(np.random.randn(len(I), len(V)),
#                   coords=[I, V],
#                   name='mu')
sigma = xr.DataArray(np.random.uniform(size=(len(x), len(y))),
                     coords=[x, y],
                     name='sigma')

# sigma = np.sqrt(variance)
z = xr.DataArray(np.linspace(0, 1, 100), dims='idx', name='z')
z = (z * sigma * 2 - sigma) * 3

norm = xr.apply_ufunc(
    stats.norm.pdf, z, 0, sigma,
    input_core_dims=[['idx'], [], []],
    output_core_dims=[['idx']],
    vectorize=True,
)
z = z.rename('z')
norm = (norm.rename('norm')
        .assign_coords(z=z))

norm
norm.hvplot.line('z', 'norm', groupby='x')
```

```python
norm.hvplot.scatter('z', 'norm', groupby='x', by='y')
```

```python
sigma.hvplot.scatter()
```

```python
def gaussian(x, mu, sigma):
        return (1/(sigma * np.sqrt(2 * np.pi))
                * np.exp( - (x - mu)**2 / (2 * sigma**2)))
```

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats as stats

mu = 0

I = xr.DataArray(np.arange(5), dims='I', name='I')
V = xr.DataArray(np.arange(5), dims='V', name='V')
# mu = xr.DataArray(np.random.randn(len(I), len(V)),
#                   coords=[I, V],
#                   name='mu')
sigma = xr.DataArray(np.random.randn(len(I), len(V)),
                     coords=[I, V],
                     name='sigma')

x = xr.apply_ufunc(
    np.linspace, mu - 3*sigma, mu + 3*sigma, 100,
    input_core_dims=[['I', 'V'], ['I', 'V'], []],
    output_core_dims=[['idx', 'I', 'V']],
)
```

```python
B = xr.DataArray(np.linspace(0, 1, 100), dims='Bidx', name='B')

norm = xr.apply_ufunc(
    gaussian, B, mu, sigma,
    input_core_dims=[['Bidx'], [], []],
    output_core_dims=[['Bidx']],
    vectorize=True,
)
B = B.rename('B')
norm = (norm.rename('norm')
        .assign_coords(B=B))
# xr.merge([norm, B])
norm
norm.hvplot.line('B', 'norm', groupby=['I'])
```
