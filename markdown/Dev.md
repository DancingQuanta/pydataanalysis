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
import scipy.stats as stats

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


x = 'idx'
y = 'x'
groupby = ['I', 'direction', 'reps']
kind = 'scatter'
plot = data.hvplot(x=x, y=y, groupby=groupby, kind=kind, dynamic=False)
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

```python
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats

import holoviews as hv
from holoviews import opts
import hvplot.pandas

hv.extension('bokeh')

# Dimensions
I = xr.DataArray(np.arange(10), dims='I', name='I')
direction = xr.DataArray(['negative', 'positive'], dims='direction', name='direction')
reps = xr.DataArray(np.arange(5), dims='reps', name='reps')
idx = np.arange(100)

# Normal distribution parameters
variance = 1
sigma = np.sqrt(variance)
mu = xr.DataArray([-1, 1], dims='direction')

# Independent variable
x = xr.DataArray(np.linspace(0, 1, 100), dims='x', name='x')
x = (x * sigma * 2 - sigma) * 3

# Generate normal distribution
y = xr.apply_ufunc(stats.norm.pdf, x, mu, sigma)

# Add noise
noise = xr.DataArray(np.random.randn(len(I), len(reps), *y.shape),
                     dims=['I', 'reps', *y.dims],
                     name='noise')

y = y + noise * 0.03
y.name = 'y'

# Assign coords
y = y.assign_coords(x=x, reps=reps, direction=direction, I=I)

def compute_marker(scatter):
    direction_markers = {
        'positive': 'circle',
        'negative': 'triangle'
    }
    sel_data = scatter.data
    direction = sel_data['direction'].iloc[0]
    return scatter.opts(marker=direction_markers[direction])

plot = (y.to_dataframe()
        .hvplot.scatter(x='x', y='y', groupby=['I', 'direction', 'reps'], dynamic=False)
        .apply(compute_marker))

plot.overlay('direction') # No display in output
plot.overlay('I') # No display in output
plot.overlay('I').layout('direction') # Only displays a bokeh toolbar
```

```python
# def apply_example(da):

#     def fn(arr):
#         first = np.asarray([1,2,5,7])
#         second = np.asarray([[3,7,1,7], [2,6,2,6]])
#         return first, second

#     results = xr.apply_ufunc(
#         fn, da,
#         input_core_dims=[['pos_ind']],
#         output_core_dims=[['seq'], ['seq1', 'seq']],
#         vectorize=True
#     )

#     var_names = ['ex1', 'ex2']
#     data_vars = {var_names[i]: results[i] for i in range(len(var_names))}
#     results = xr.Dataset(data_vars)
#     return results
# apply_example(arr)
```

```python
# import numpy as np
# import xarray as xr
# from scipy.signal import find_peaks

# # Generate waveform
# x = (np.sin(2*np.pi*(2**np.linspace(2,10,1000))*np.arange(1000)/48000)
#      + np.random.normal(0, 1, 1000) * 0.15)

# # Find peaks non-xarray way
# peaks, _ = find_peaks(x, prominence=1)
# print(peaks)

# # Cast waveform to xr.DataArray
# x = xr.DataArray(x, dims='time')

# # Duplicate data along a new dimension
# rep = xr.DataArray(range(11), dims='repeat')
# x = (x.broadcast_like(rep).assign_coords(repeat=rep))

# def process_peaks(arr):
#     # Apply find_peaks

#     # xr.apply_ufunc passes along a read-only buffer source array.
#     # Functions in this applying function do not work with read-only array
#     # So make a copy
#     arr = arr.copy()

#     # Finally execute find_peaks
#     peaks, _ = find_peaks(arr, prominence=1)
#     return peaks

# # Apply function to array
# results = xr.apply_ufunc(
#     process_peaks, x,
#     input_core_dims=[['time']],
#     output_core_dims=[['peaks']],
#     vectorize=True
# )

# # Should show repeats of peak results
# print(results)
```

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

I = xr.DataArray(np.arange(10), dims='I', name='I')
V = xr.DataArray(np.arange(5), dims='V', name='V')
direction = xr.DataArray(['negative', 'positive'], dims='direction', name='direction')
# mu = xr.DataArray(np.random.randn(len(I), len(V)),
#                   coords=[I, V],
#                   name='mu')
sigma = xr.DataArray(np.random.randn(len(I), len(V), len(direction)),
                     coords=[I, V, direction],
                     name='sigma')

# sigma = np.sqrt(variance)
B = xr.DataArray(np.linspace(0, 1, 100), dims='Bidx', name='B')
B = (B * sigma * 2 - sigma) * 3

norm = xr.apply_ufunc(
    stats.norm.pdf, B, 0, sigma,
    input_core_dims=[['Bidx'], [], []],
    output_core_dims=[['Bidx']],
    vectorize=True,
)
B = B.rename('B')
norm = (norm.rename('norm')
        .assign_coords(B=B))
# xr.merge([norm, B])
norm
norm.hvplot.line('B', 'norm', groupby=['I', 'V'])
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

I = xr.DataArray(np.arange(10), dims='I', name='I')
V = xr.DataArray(np.arange(5), dims='V', name='V')
direction = xr.DataArray(['negative', 'positive'], dims='direction', name='direction')
# mu = xr.DataArray(np.random.randn(len(I), len(V)),
#                   coords=[I, V],
#                   name='mu')
sigma = xr.DataArray(np.random.randn(len(I), len(V), len(direction)),
                     coords=[I, V, direction],
                     name='sigma')

x = xr.apply_ufunc(
    np.linspace, mu - 3*sigma, mu + 3*sigma, 100,
#     input_core_dims=[[], [], []],
#     output_core_dims=[[]],
)
x
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
norm.hvplot.line('B', 'norm', groupby=['I', 'V'])
```

```python
params = {
    'text.latex.preamble': [r'\usepackage{gensymb}'],
#     'axes.labelsize': 8, # fontsize for x and y labels (was 10)
#     'axes.titlesize': 8,
#     'font.size': 8, # was 10
#     'legend.fontsize': 8, # was 10
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
    'text.usetex': True,
    #'figure.figsize': [fig_width, fig_height],
    'font.family': 'serif',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'errorbar.capsize': 2,
    'axes.axisbelow': True,
}
import matplotlib
matplotlib.rcParams.update(params)

norm.sel(I=0)
one = norm.sel(V=0).hvplot.line('B', 'norm', dynamic=False)
two = norm.sel(V=1).hvplot.line('B', 'norm', dynamic=False)
layout = one + two
(layout.opts(fig_size=100))
```

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats as stats

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

```python

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats as stats

I = xr.DataArray(np.arange(10), dims='I', name='I')
V = xr.DataArray(np.arange(5), dims='V', name='V')
# mu = xr.DataArray(np.random.randn(len(I), len(V)),
#                   coords=[I, V],
#                   name='mu')
sigma = xr.DataArray(np.random.randn(len(I), len(V)),
                     coords=[I, V],
                     name='sigma')

# sigma = np.sqrt(variance)
B = xr.DataArray(np.linspace(0, 1, 100), dims='Bidx', name='B')
B = (B * sigma * 2 - sigma) * 3

norm = xr.apply_ufunc(
    stats.norm.pdf, B, 0, sigma,
    input_core_dims=[['Bidx'], [], []],
    output_core_dims=[['Bidx']],
    vectorize=True,
)
B = B.rename('B')
norm = (norm.rename('norm')
        .assign_coords(B=B))
# xr.merge([norm, B])
norm
norm.hvplot.line('B', 'norm', groupby=['I', 'V'])
```

```python
# def f(x, xb, y0, ks):
#     if len(xb) != len(ks)+1:
#         raise

#     yb = np.like(xb)
#     yb[0] = y0
#     for i in range(len(yb)-1)
#         yb[i+1] = yb[i] + (ks[i+1] - xb[i])

#     for i in range(len(xb)):
#         if i == 0:
#             cond = x < xb[i]
#             y = yb[i] + ks[i]
#         elif i == len(xb):
#             cond = x >= xb[i]
#         else:
#             cond = (x >= xb[i-1] & x < xb[i])



```

```python
# from scipy import optimize
# import matplotlib.pyplot as plt
# import numpy as np

# sel_data = data2.sel(I=-2, rep=4, direction='negative')
# x = sel_data['B'].values
# y = sel_data['sig1'].values

# verts = [89, 93, 158, 167]
# xn = x[verts]
# y0 = y[verts[0]]
# kn = [1] * 5
# p0 = xn + [y0] + kn

# def f(x,x0,x1,x2,x3,x4,y0,
#       k0,k1,k2,k3,k4,k5):
#     # x0,y0 : first breakpoint
#     # x1 : second breakpoint
#     # k1,k2,k3 : 3 slopes.

#     y1 = y0 + k1*(x1-x0)
#     y2 = y1 + k2*(x2-x1)
#     y3 = y2 + k3*(x3-x2)
#     y4 = y3 + k4*(x4-x3)
#     return (
#     (x<x0)              *   (y0 + k0*(x-x0))      +
#     ((x>=x0) & (x<x1))  *   (y0 + k1*(x-x0))      +
#     ((x>=x1) & (x<x2))  *   (y1 + k2*(x-x1))      +
#     ((x>=x2) & (x<x3))  *   (y2 + k3*(x-x2))      +
#     ((x>=x3) & (x<x4))  *   (y3 + k4*(x-x3))      +
#     (x>=x4)             *   (y4 + k5*(x-x4)))

# p, e = optimize.curve_fit(f, x, y,p0)
# print(p)
# print(e)
# plt.plot(x, y, "o")
# plt.plot(x,f(x,*p))
```

```python
# x = np.array([-1.280199006, -1.136209343, -1.048070216, -0.9616764178, -0.8752826199, -0.7871434926, -0.6981317008, -0.6108652382, -0.5235987756, -0.4372049776,
#  -0.3490658504, -0.2644173817, -0.1762782545,
# -0.0907571211, 0, 0.09250245036, 0.1762782545, 0.2661627109, 0.3516838443, 0.4345869837, 0.529707428, 0.6108652382, 0.7007496947, 0.7880161573, 0.872664626,
# 0.9616764178, 1.055051533, 1.160643953, 1.274090354, 1.413716694])

# y = np.array([-0.05860218717, -0.05275174988, -0.04961805822, -0.02860635697, -0.04150466841, -0.02672933264, -0.02422597285, -0.03056176732, -0.02885180089, -0.02085851636,
# -0.02873319291, -0.02374542821, -0.02132671806,
# -0.02088924602, -0.0216617248, -0.01835553738, -0.01369531698, -0.01331112368, -0.01156455074, -0.009163690404, -0.003542622659, -0.003515924976, -0.003828831726, -0.002622163805, -0.001622083468,
# -0.00297346133, -0.001845415856, -0.001913228234, -0.001495496086, -0.001454621173])

# def f(x,x0,y0,x1,k1,k2,k3):
#     # x0,y0 : first breakpoint
#     # x1 : second breakpoint
#     # k1,k2,k3 : 3 slopes.

# #     print((x<x0)   *   (y0 + k1*(x-x0)))

#     y1=y0+ k2*(x1-x0) # for continuity
#     return (
#     (x<x0)              *   (y0 + k1*(x-x0))      +
#     ((x>=x0) & (x<x1))  *   (y0 + k2*(x-x0))      +
#     (x>=x1)             *   (y1 + k3*(x-x1)))

# p0=(-.7,-0.03,.5,0.03,0.02,0.01)
# p , e = optimize.curve_fit(f, x, y,p0)
# print(p)
# plt.plot(x, y, "o")
# plt.plot(x,f(x,*p))
```

```python
# from scipy import optimize
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
# y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

# def piecewise_linear(x, x0, y0, k1, k2):
#     return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

# p , e = optimize.curve_fit(piecewise_linear, x, y)
# xd = np.linspace(0, 15, 100)
# plt.plot(x, y, "o")
# plt.plot(xd, piecewise_linear(xd, *p))
# print(p)
```

```python
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
# from scipy.signal import find_peaks

# sel_data = sel_data.sel(I=-2, rep=4, direction='negative')
# x = sel_data['B'].values
# y = sel_data['sig1'].values

# tck = interpolate.splrep(x, y, k=3, s=0.000000002)
# xnew = np.arange(x[0], x[-1], 0.1)

# fig, axes = plt.subplots(3)

# axes[0].plot(x, y, 'x', label = 'data')
# axes[0].plot(xnew, interpolate.splev(xnew, tck, der=0), label = 'Fit')
# axes[1].plot(x, interpolate.splev(x, tck, der=1), label = '1st dev')
# dev_2 = interpolate.splev(x, tck, der=2)
# axes[2].plot(x, dev_2, label = '2st dev')

# turning_point_mask = dev_2 == np.amax(dev_2)
# peaks, _ = find_peaks(dev_2)
# print(peaks)
# axes[2].plot(x[peaks], dev_2[peaks],'rx',
#              label = 'Turning point')
# peaks, _ = find_peaks(dev_2*-1)
# print(peaks)
# axes[2].plot(x[peaks], dev_2[peaks],'bx',
#              label = 'Turning point')
# for ax in axes:
#     ax.legend(loc = 'best')
```

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
