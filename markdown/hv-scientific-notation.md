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
from holoviews import opts
import hvplot.xarray

hv.extension('matplotlib')
hv.notebook_extension(display_formats=['html', 'svg'])
hv.output(fig='svg')

# Dimensions
reps = xr.DataArray(np.arange(5), dims='reps', name='reps')
idx = np.arange(600)

x = xr.DataArray(np.random.randn(len(idx), len(reps)),
                 dims=['idx', 'reps'],
                 name='x')
y = xr.DataArray(np.random.randn(len(idx), len(reps)),
                 dims=['idx', 'reps'],
                 name='y')
y = y * 1e-6

# Merge x, y
data = xr.merge([x, y])

# Assign coords
data = data.assign_coords(reps=reps)

def parse_label(s):
    lst = (s.split(' ('))
    if len(lst) == 1:
        return lst[0], ""
    elif len(lst) == 2:
        return lst[0], lst[1].rstrip(')')

def sciticks(plot, element):
    """
    https://www.programcreek.com/python/example/102352/matplotlib.pyplot.ticklabel_format
    https://werthmuller.org/blog/2014/move-scientific-notation/
    http://greg-ashton.physics.monash.edu/setting-nice-axes-labels-in-matplotlib.html
    https://stackoverflow.com/questions/41752111/how-to-control-holoviews-y-tick-format
    https://peytondmurray.github.io/coding/fixing-matplotlibs-scientific-notation/#
    """
    ax = plot.handles['axis']
    print(element.data)

    # Set sci notation
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fmt = ax.yaxis.get_major_formatter()
    offset = fmt.get_offset()
    print(fmt.get_useMathText())
    print(offset)

    # Get label
    label = ax.get_ylabel()
    label, unit = parse_label(label)

    # Construct unit
    unit_lst = []

    if offset:
        unit_lst.append(offset)

    if unit:
        unit_lst.append(unit)

    if len(unit_lst) > 0:
        unit = ' '.join(unit_lst)

        label = f'{label} ({unit})'

    # Set label
    print(label)

data.hvplot.scatter(x='x', y='y', groupby=['reps'], dynamic=False)
```
