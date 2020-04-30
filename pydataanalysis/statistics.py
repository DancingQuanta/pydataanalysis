import numpy as np
import pandas as pd
import xarray as xr
import scipy
import uncertainties
from uncertainties import ufloat, unumpy as unp
# import metrolopy as uc
# from .data import *
from .xarray_utils import *
from .utils import sqrt_ss

def zscore(data, dim):
    """
    Calculate statistics with central tendency (mean) and 
    dispersion (std) metrics and z-score from these metrics
    
    Params:
        data: Data to be operated on
        dim: Dimension to reduce
    """
    # Get attrs
    attrs = data.attrs
    
    # Calculate central and dispersion
    mean = data.mean(dim=dim, skipna=True)
    std = data.std(dim=dim, skipna=True)
    
    # Calculate zscore from central and dispersion
    zscores = ((data - mean) / std)
    
    # Combine into new uncertainties uarray
    data = xr.Dataset({'mean': mean,
                       'std': std,
                       'zscores': zscores})
    
    # Resture attrs
    data.attrs = attrs
    
    return data

def mean_confidence(ds, k=1):
    
    lower = ds['mean'] - k * ds['std']
    upper = ds['mean'] + k * ds['std']
    
    ds = ds.assign({f"{k} lower": lower, f"{k} upper": upper})
    return ds

def mean_confidence(ds, k=1):
    
    lower = ds['Measur'] - k * ds['std']
    upper = ds['mean'] + k * ds['std']
    
    ds = ds.assign({f"{k} lower": lower, f"{k} upper": upper})
    return ds

def robust_zscore(data, dim):
    """
    Calculate statistics with central tendency (median) and
    dispersion (median absolute deviation) metrics and
    z-score from these metrics
    
    Params:
        data: Data to be operated on
        dim: Dimension to reduce
    """
    # Get attrs
    attrs = data.attrs
    
    # Transpose reducing dimension to front
    data = data.transpose(dim, ...)
    
    # Calculate central and dispersion
    median = data.median(dim=dim, skipna=True)
    
    # Testing
    def fn(*args, **kwargs):
#         print(*args)
        out = scipy.stats.median_absolute_deviation(*args, **kwargs)
#         print(out)
        return out
        
    mad = xr.apply_ufunc(
        fn, data,
#         scipy.stats.median_absolute_deviation, data,
        kwargs={'nan_policy' :'omit'},
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True
    )
    
    # Calculate zscore from central and dispersion
    zscores = ((data - median) / mad)
    
    # Combine into new uncertainties uarray
    data = xr.Dataset({'median': median,
                       'mad': mad,
                       'rzscores': zscores})
    
    # Resture attrs
    data.attrs = attrs
    
    return data

def IQR_stats(data, dim):
    """
    Calculate quantiles 1 and 3 and the difference between them (IQR)
    
    Params:
        data: Data to be operated on
        dim: Dimension to reduce
    """
    # Get attrs
    attrs = data.attrs
    
    # Calculate quantiles
    Q1 = data.quantile(0.25, dim=dim)
    Q3 = data.quantile(0.75, dim=dim)
    
    # Interquantile range
    IQR = Q3 - Q1
    
    # Combine into new uncertainties uarray
    coords = xr.IndexVariable('stats', ['Q1', 'Q3', 'IQR'])
    data = xr.concat([Q1, Q3, IQR], coords)
    
    # Resture attrs
    data.attrs = attrs
    
    return data

def IQR_outlier(ds, dim):
    
    stats = IQR_stats(ds, dim)
    
    Q1 = stats.sel(stats='Q1')
    Q3 = stats.sel(stats='Q3')
    IQR = stats.sel(stats='IQR')
    
    cond = ~((ds < (Q1 - 1.5 * IQR)) |
             (ds > (Q3 + 1.5 * IQR))).any()
    
    ds = ds.where(cond)
    return ds, stats
    
def xstatistics(data, dim): 
    
    # Statistics
    zscore_data = zscore(data, dim)
    rzscore_data = robust_zscore(data, dim)
    
    stats = xr.merge([data, zscore_data, rzscore_data])
    return stats

def xstats(data, dim):
    # Get attrs
    attrs = data.attrs
    
    # Calculate mean and sem
    measurand = data.mean(dim=dim, skipna=True)
    uncertainty = data.std(dim=dim, skipna=True)
#     uncertainty = data.reduce(sem, dim=dim)
    
    # Combine into new uncertainties uarray
    coords = xr.IndexVariable('unc', ['Measurand', 'Uncertainty'])
    data = xr.concat([measurand, uncertainty], coords)
    
    # Resture attrs
    data.attrs = attrs
    
    return data

def stats2unc_dim(ds):
    if 'stderr' in ds:
        central = 'value'
        dispersion = 'stderr'
    elif 'std' in ds:
        central = 'mean'
        dispersion = 'std'
    
    # Calculate mean and sem
    measurand = ds[central]
    uncertainty = ds[dispersion]
    
    # Combine into new uncertainties uarray
    coords = xr.IndexVariable('unc', ['Measurand', 'Uncertainty'])
    da = xr.concat([measurand, uncertainty], coords)
    
    return da

def filter_uncertainty(ds, variable, level):
    ds[variable + '_relative'] = (ds[variable + '_unc']
                                  / ds[variable])*100
    cond = ds[variable + '_relative'] > level
    ds[variable + '_unc'] = ds[variable + '_unc'].where(cond)
    return ds

def extract_measurand(da):
    if 'unc' in da.dims:
        return da.sel(unc='Measurand')
    else:
        try:
            return xr.apply_ufunc(unp.nominal_values, da, keep_attrs=True)
        except:
            return da

def extract_unc(da):
    if 'unc' in da.dims:
        return da.sel(unc='Uncertainty')
    else:
        try:
            return xr.apply_ufunc(unp.std_devs, da, keep_attrs=True)
        except:
            return da

        
def unc_dim_to_type(da):
    
    # If variable have a dimension 'unc' convert it to uarray
    # else return unchanged
    if 'unc' in da.dims:
        return xr.apply_ufunc(unp.uarray,
                              da.sel(unc='Measurand'),
                              da.sel(unc='Uncertainty'),
                              keep_attrs=True)
    else:
        return da
        
def unc_dim_to_vars(ds):
    
    if not hasattr(ds, 'data_vars'):
        ds = ds.to_dataset()

    # Preserve attrs
    attrs = ds.attrs
    
    # Split dataset so we can work on variables with uncertainty separately
    unc_names = []
    non_unc_names = []
    for da_name, da in ds.data_vars.items():
        # If variable have a dimension 'unc' add to a list 
        # otherwise add to other list
        if 'unc' in da.dims:
            unc_names.append(da_name)
        else:
            non_unc_names.append(da_name)
    
    # Pull out variables
    unc_ds = ds[unc_names]
    non_unc_ds = ds[non_unc_names]
    
    # Dissolve uncertainties
    measurand_ds = unc_ds.sel(unc='Measurand', drop=True)
    unc_ds = unc_ds.sel(unc='Uncertainty', drop=True)
    
    # Rename variables in unc dataset to include '_unc'
    # Generate dictionary mapping of names to names with '_unc' appended.
    name_map = {name: name + '_unc' for name in list(unc_ds.keys())}
    
    # Rename variables
    unc_ds = unc_ds.rename(name_map)
    
    # Merge measurand with uncertainty data as separate variables
    ds = xr.merge([measurand_ds, unc_ds, non_unc_ds])
    
    # Restore attrs
    ds.attrs = attrs
    
    return ds
        
def unc_type_to_dim(da):
    # Transform uncertainties to dimension
    
    # The commented out block fails because np function can't work on
    # uncertainties type without its dedicated unumpy suite.
    # This give me an idea to ram the data though a numpy function to
    # check whether it works. If it does not work then it is likely to
    # be a uncertainties type
#
    # Extract components
    try:
        # Make sure this fails to prove the input is of uncertainties type
        mask = np.isnan(da)
        raise ValueError('Not an uncertainties type')
    except TypeError:
        nominal = xr.apply_ufunc(unp.nominal_values, da, keep_attrs=True)
        unc = xr.apply_ufunc(unp.std_devs, da, keep_attrs=True)
    
    # Concat into one dataset along a new dimension
    nominal = nominal.assign_coords(unc='Measurand')
    unc = unc.assign_coords(unc='Uncertainty')
    da = xr.concat([nominal, unc], 'unc')
    return da
    
def split_uncertainty(ds):
    
    if not hasattr(ds, 'data_vars'):
        ds = ds.to_dataset()
    
    def apply_func(da):
        try:
            return unc_type_to_dim(da)
        except ValueError:
            return da
    
    ds = ds.map(apply_func, keep_attrs=True)
    
    return ds

def combine_uncertainty(ds):
    """
    Convert DataArray or DataSet with variables with dimension 'unc' 
    to uncertainty type
    """
    
    # Check if input is a DataArray 
    if not hasattr(ds, 'data_vars'):
        ds = unc_dim_to_type(ds)
    else:
        ds = ds.map(unc_dim_to_type, keep_attrs=True)
    
    return ds
    
def confidence_intervals(ds, k=1):
    lower = ds.sel(unc='Measurand') - k * ds.sel(unc='Uncertainty')
    lower = lower.assign_coords(unc=f"{k} lower")
    upper = ds.sel(unc='Measurand') + k * ds.sel(unc='Uncertainty')
    upper = upper.assign_coords(unc=f"{k} upper")
    ds = xr.concat([ds, lower, upper], 'unc')
    return ds
    
def xtable_uncertainty(ds, index_name=''):
    
    # Drop coords
    ds = ds.reset_coords(drop=True)
    
    # Get attrs
    attrs = get_attrs(ds)

    # Calculate relative uncertainty
    relative = np.abs(ds.sel(unc='Uncertainty') / ds.sel(unc='Measurand'))

    # add relative uncertainty to dataset
    ds = xr.concat([ds, relative.assign_coords(unc='Relative uncertainty')], 'unc')

    # Restore attrs to dataset
    restore_attrs(ds, attrs)
    
    # Convert to dataframe
    table = ds.to_dataframe()

    # Use long_name and units for variables
    table.columns = [pprint_label(ds[var]) for var in table.columns]

    # Clear index name and transpose
    table.index.name = ''
    table = table.T

    table.index.name = index_name
    return table

def pdtable_uncertainty(df):
    ds = scalar_to_xarray(df)
    ds = split_uncertainty(ds)
    return xtable_uncertainty(ds)

def mul_prog(X, nominal, unc):
    """Error propagation for multiplication or division"""
    rel = unc / nominal
    return X * sqrt_ss(rel)
