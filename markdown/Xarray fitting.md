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

# Applying fitting functions to xarray objects

Apply a function to a dataset over multiple dimensions.


## Broadcast

In order to apply data along one dimension, broadcast the coordinate of this dimension across other dimensions.


### Generate linear and noisy data

```python

```

### Apply Statsmodules.OLS to the data

```python
# linear regression of y against x

# Prepare
pre_fit = (data
           .swap_dims({'temp': 'temp_index'})
           .pipe(lambda x: x.assign_coords({'temp': x.temp.broadcast_like(x)}))
           .stack(obs=['temp_index', 'point', 'direction'])
           .dropna(dim='obs')
           .pipe(unc_dim_to_vars))

# Fit
fitted_data = pre_fit.pipe(xsm_ols, 'temp', 'bridge', 'obs')
```

## Groubpy

```python
from src.model_fitting import 

# Prepare feature function
def func_features(df):
    df['T_0'] = 1.
    df['T_1'] = df['temp']

    return df[['T_1', 'T_0']]

# Fit
def fn(da):
    da = (da.groupby('bias')
          .map(process_sm_wls, args=([func_features]),
               y='bridge', yerr='bridge_unc'))
    return da

fitted_data = (data
               .pipe(unc_dim_to_vars)
               .groupby('mode')
               .map(fn))
```
