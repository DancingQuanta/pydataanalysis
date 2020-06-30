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

# Groupby apply over a number of dimensions

Apply a function to a groupby over multiple dimensions.


## First case

Apply groupby over multiple dimensions

```python
import numpy as np
import xarray as xr

# Dimensions
N = xr.DataArray(np.arange(100), dims='N', name='N')
reps = xr.DataArray(np.arange(5), dims='reps', name='reps')
horizon = xr.DataArray([1, -1], dims='horizon', name='horizon')
horizon.attrs = {'long_name': 'Horizonal', 'units': 'H'}
vertical = xr.DataArray(np.arange(1, 4), dims='vertical', name='vertical')
vertical.attrs = {'long_name': 'Vertical', 'units': 'V'}

# Variables
x = xr.DataArray(np.random.randn(len(N), len(reps), len(horizon), len(vertical)),
                 dims=['N', 'reps', 'horizon', 'vertical'],
                 name='x')
y = x * 0.1
y.name = 'y'

# Merge x, y
data = xr.merge([x, y])

# Assign coords
data = data.assign_coords(reps=reps, vertical=vertical, horizon=horizon)

# Function that stack all but one diensions and groupby over the stacked dimension.
def process_multi_groupby(ds, dim, func, *args):

    # Function to apply to stacked groupby
    def apply_fn(ds, dim, func, *args):
        print(ds)
        print(dim)
        return ds

    # Get list of dimensions
    groupby_dims = list(ds.dims)

    # Remove dimension not grouped
    groupby_dims.remove(dim)

    # Groupby and apply
    ds2 = ds.groupby(groupby_dims, squeeze=False).map(apply_fn, args=(dim, func, *args))

    # Restore attrs
    for dim in groupby_dims:
        ds2[dim].attrs = ds[dim].attrs

    return ds2

# Function to apply on groupby
def fn(ds):
    return ds

# Run groupby with applied function
data.pipe(process_multi_groupby, 'N', fn)
```

## Second case

Since it is not possible to groupby multiple dimensions directly a workaround is needed.

To groupby multiple dimensions is to

1. Stack them into one dimension and then groupby over that dimension
2. Apply a function to the groupby object which should unstack the stacked dimension

The implementation steps are:

1. stack chosen dimensions
2. groupby the stacked dimension
3. apply a wrapper function on each group
  1. The wrapper function passes the data with unstacked group coord to the applied function 
  2. Add multi-index stacked group coord back to the results of this function
4. combine the groups
5. Unstack stacked dimension

```python
import numpy as np
import xarray as xr

# Dimensions
N = xr.DataArray(np.arange(100), dims='N', name='N')
z = xr.DataArray([1, -1], dims='z', name='z')
reps = xr.DataArray(np.arange(5), dims='reps', name='reps')

# Variables
x = xr.DataArray(np.random.randn(len(N), len(z), len(reps)),
                 dims=['N', 'z', 'reps'],
                 name='x')
y = x * 0.1
y.name = 'y'

# Merge x, y
data = xr.merge([x, y])

# Assign coords
data = data.assign_coords(z=z, reps=reps)

# Function that stack all but one diensions and groupby over the stacked dimension.
def process_stacked_groupby(ds, dim, func, *args):

    # Function to apply to stacked groupby
    def apply_fn(ds, dim, func, *args):

        # Get groupby dim
        groupby_dim = list(ds.dims)
        groupby_dim.remove(dim)
        groupby_var = ds[groupby_dim]

        # Unstack groupby dim
        ds2 = ds.unstack(groupby_dim).squeeze()

        # perform function
        ds3 = func(ds2, *args)

        # Add mulit-index groupby_var to result
        ds3 = (ds3
               .reset_coords(drop=True)
               .assign_coords(groupby_var)
               .expand_dims(groupby_dim)
             )
        return ds3

    # Get list of dimensions
    groupby_dims = list(ds.dims)

    # Remove dimension not grouped
    groupby_dims.remove(dim)

    # Stack all but one dimensions
    stack_dim = '_'.join(groupby_dims)
    ds2 = ds.stack({stack_dim: groupby_dims})

    # Groupby and apply
    ds2 = ds2.groupby(stack_dim, squeeze=False).map(apply_fn, args=(dim, func, *args))

    # Unstack
    ds2 = ds2.unstack(stack_dim)

    # Restore attrs
    for dim in groupby_dims:
        ds2[dim].attrs = ds[dim].attrs

    return ds2

# Function to apply on groupby
def fn(ds):
    return ds

# Run groupby with applied function
data.pipe(process_stacked_groupby, 'N', fn)
```

```python

```
