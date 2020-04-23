import numpy as np
import xarray as xr
from .pandas_utils import pd
from .utils import magnitude_thousand

xr.set_options(keep_attrs=True)

def get_attrs(ds):
    attrs = {}
    for var in ds.variables:
        attrs[var] = ds[var].attrs
    return attrs

def restore_attrs(ds, attrs):
    for var in ds.variables:
        if var in attrs:
            ds[var].attrs = attrs[var]

def apply_ufunc_to_dataset(func, *args, names, **kwargs):
    results = xarray.apply_ufunc(func, *args, **kwargs)
    return xarray.Dataset(dict(zip(names, results))) 

def xmagnitude_thousand(da):
    # Get max
    number = da.max()

    # Calculate thousand muliplter
    exponent = magnitude_thousand(number)

    # Adjust scale
    if exponent != 0:
        units = da.attrs.get('units', '')

        da = da / 10**exponent

        da.attrs['units'] = rf'{units} $\times 10^{{{exponent}}}$'

    return da

def pprint_label(da):
    # Shorthand label
    label = da.name
    
    # long name
    if 'long_name' in da.attrs:
        label = da.attrs['long_name']
    
    # Units
    if 'units' in da.attrs:
        label = label + ' ({units})'.format(**da.attrs)
    
    return label

def pprint_value(da):
    # Convert value to string
    value = da.item()
    if isinstance(value, int):
        value = str(value)
    elif isinstance(value, float):
        value = F"{value:.2f}"
    
    # Add unit if available
    if 'units' in da.attrs:
        value = value + ' ' + da.attrs['units']
    
    return value
        
def pprint_label_value(da):
    # Pretty print value
    value = pprint_value(da)
    
    # shorthand label
    label = da.name
    
    # Build string label
    label = f"{label}: {value}"
    
    return label

# def pprint_title(da):
#     value = str(da.item())
#     if 'units' in da.attrs:
#         value = value + ' ' + da.attrs['units']
#     return f'{da.name}: {value}'

def sel_rename(ds, dim, old_var, new_var):
    return ds.sel({dim: new_var}, drop=True).rename({old_var: new_var})

def replace_var(ds, old, new):
    ds = ds.copy(deep=True)
    ds[new] = ds[old]
    ds = ds.drop_vars(old)
    return ds

def normalize(arr, axis):
    min = np.min(arr, axis)
    return (arr - min)/(np.max(arr, axis) - min)

def flip(arr):
    return (arr - np.max(arr)) * -1

def reject_zero(ds, dim):
    ds = ds.where(ds[dim]!=0, drop=True)
    return ds
    
def multiply(da, f):
    func = lambda a, b: a * b
    da = xr.apply_ufunc(func, da, f, keep_attrs=True)
    return da
    
def sign_flip_dim(ds, var, dim, flip=False):
    # Sign
    s = np.sign(ds[dim])
    
    # Reverse sign
    if flip:
        s = s * -1

    # Add sign info to dataset
    ds = ds.assign_coords(s=s)
    
    # Multiply by sign
    ds[var] = multiply(ds[var], ds['s'])
    
#     ds[var] = xr.apply_ufunc(lambda a, b: a * b, ds[var], ds['s'],
#                              keep_attrs=True)
    
    return ds

def sign_flip_coord(da, s, reverse=False):
    
    # Reverse sign
    if reverse:
        s = s * -1
    
    # Multiply by sign
    da = multiply(da, s)
    
    # Annotate negative sign
    lst = []
    for i, v in enumerate(s):
        k = da[s.name][i].item()
        if v.item() == -1:
            k = rf"{k} $\times {{-1}}$"
        lst.append(k)
    da[s.name] = lst

    return da

def reverse_sign_flip(ds, var):
    ds = ds.copy(deep=True)
    
    ds[var] = multiply(ds[var], ds['s'])
#     ds[var] = xr.apply_ufunc(lambda a, b: a * b, ds[var], ds['s'],
#                              keep_attrs=True)
#     ds[var] = ds[var] * ds['s']
    return ds

def dropna(*args):
    # Mask based on nan
    masks = ~np.isnan(args)
    
    # Reduce by AND
    mask = np.logical_and.reduce(masks)
    
    # reindex with mask
    out = []
    for arg in args:
        out.append(arg[mask])
    
    return out

def dropna_by_var(ds, var, dim):
    var = ds[var].dropna(dim=dim)
    ds = ds.sel({dim: var[dim]})
    return ds

def scalar_to_xarray(data):
    index = ['long_name', 'units', 'value']
    data = pd.DataFrame(data, index=index).to_xarray()
    def fn(da):
        da = da.to_dataset(dim='index')
        value = da['value']
        value.attrs = {'long_name': da['long_name'].item(),
                       'units': da['units'].item()}
        return value
    data = data.apply(fn)
    return data

def select_uniques(ds, dim):
    # Find index of unique
    _, index = np.unique(ds[dim], return_index=True)
    return ds.isel({dim: index})

def mask_nonuniques(ds, dim):
    # Find index of unique
    _, index = np.unique(ds[dim], return_index=True)
    mask_array = xr.zeros_like(ds[dim], dtype=bool) 
#     mask_array = np.zeros(len(ds[dim]), dtype=bool) 
    mask_array[index] = True
    return ds.where(mask_array)

def stack_str(ds, dims):
    # String separator
    sep = ', '
    
    # Add units
    ds2 = ds.copy(deep=True)
    for dim in dims:
        ds2[dim] = [pprint_value(v) for v in ds2[dim]]
        
    # Stack dim and drop nan
    stack_value_dim = sep.join(dims)
    stack_dim = f"{stack_value_dim} stack"
    
    ds2 = ds2.stack({stack_dim: dims}).dropna(dim=stack_dim, how='all')
    ds = ds.stack({stack_dim: dims}).dropna(dim=stack_dim, how='all')
    
    # Generate new dimension labels
    lst = [sep.join(map(str, vs)) for vs in ds2.indexes[stack_dim].tolist()]
    
    # Add the new labels as a new variable
    ds = ds.assign_coords({stack_value_dim: xr.DataArray(lst, dims=stack_dim)})
    
    return ds

def arange(start, stop, axis=0):
    arr = np.arange(start, stop)
    return arr

def xarange(start, stop, dim):
    da = xr.apply_ufunc(
        arange, start, stop,
        # input_core_dims=[[], []],
        output_core_dims=[['seq', dim]],
        # vectorize=True
    )
    return da

def separate_regions(a, m):
    m0 = np.concatenate(( [False], m, [False] ))
    idx = np.flatnonzero(m0[1:] != m0[:-1])
    return [a[idx[i]:idx[i+1]] for i in range(0,len(idx),2)]

def mask_start_stop(a, trigger_val=True):
    """
    Find edges in a mask
    """
    # "Enclose" mask with sentients to catch shifts later on
    mask = np.r_[False,np.equal(a, trigger_val),False]

    # Get the shifting indices
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    # Get the start and end indices with slicing along the shifting ones
    return zip(idx[::2], idx[1::2]-1)

def mask_block_edge(mask, max_len):
    # Indices of edges of regions that re monotonically decreasing
    inner_edge_idx = list(mask_start_stop(mask))
    
    # Build indices of start and stop of adjacent regions
    edge_idx = []
    
    block_idx = 0
    # Is first block start of array?
    edge = inner_edge_idx[0]
    start, stop = edge
    if start != 0:
        stop = start - 1
        start = 0
        edge_idx.append((start, stop))
        block_idx = 1
    edge_idx.append(edge)
    for edge in inner_edge_idx[1:]:
        # What are previous block edge indices?
        prev = edge_idx[block_idx]
        
        # Calulate start and stop of a block between 
        # last block and this block
        _, start = prev
        stop, _ = edge
        new_block = start+1, stop-1
        
        # Append last block to edge_idx
        edge_idx.append(new_block)
        
        # Append current block to edge_idx
        edge_idx.append(edge)
        
        # Increment by 2
        block_idx = block_idx + 2
    
    # Last block
    edge = inner_edge_idx[0]
    start, stop = edge
    if stop != max_len:
        start = stop + 1
        stop = max_len - 1
        edge_idx.append((start, stop))

    return edge_idx

def monotonic_array(arr):
    ## Find monotonically decreasing regions
    
    # Mark regions of higher value than right neighbours
    # Neighbours that change from True to False are
    # changing direction from negative to positive
    # otherwise False to True changing direction from
    # positive to negative
    mask = arr[:-1] > arr[1:]
    segment_idx = mask_block_edge(mask, len(arr))
    return segment_idx

def isolate_dim(ds, dim):
    
    # Split dataset so we can work on variables with dim separately
    dim_da_list = []
    non_dim_da_list = []
    for da_name, da in ds.data_vars.items():
        # If variable have a dimension 'dim' add to a list 
        # otherwise add to other list
        if dim in da.dims:
            dim_da_list.append(da_name)
        else:
            non_dim_da_list.append(da_name)
            
    # Extract datasets
    dim_ds = ds[dim_da_list]
    
    if non_dim_da_list:
        non_dim_ds = ds[non_dim_da_list]
        return dim_ds, non_dim_ds
    else:
        return dim_ds, None

def split_coords(ds, coords, dim, region_points, new_dim):
    """
    Split a coordinate across a new dimension
    """

    # edge_dim = 'idx'
    # region_points = {k : (edge_dim, v) for k, v in region_points.items()}
    # da = xr.Dataset(region_points).to_array(dim=new_dim, name=dim)
    # da = xarange(da.isel(idx=0), da.isel(idx=1), new_dim)
    # lst = [np.arange(da.isel({'idx': 0, new_dim: k}),
    #                  da.isel({'idx': 1, new_dim: k}))
    #        for k in range(len(da[new_dim]))]
    # da = xr.concat(lst, new_dim)
    # da = xr.apply_ufunc(
    #     np.arange, da.isel(idx=0), da.isel(idx=1),
    #     input_core_dims=[[new_dim], [new_dim]],
    #     output_core_dims=[[edge_dim, new_dim]],
    #     vectorize=True
    # )

    # print(da)
    # da = ds.sel(field_index=slice(da.isel(idx=0), da.isel(idx=1)))
    # print(da)
    # raise(da)

    # split dataset
    dim_ds, non_dim_ds = isolate_dim(ds, dim)

    # Loop over each new_dim to construct new dimension
    new_dim_list = []
    for k, v in region_points.items():
        # Select part of the cycle
        slice_new_dim = dim_ds.sel({dim: slice(*v)})

        # Change dimension coords from dim to coords
        # and assign new_dim coordinate
        slice_new_dim = (slice_new_dim
                         .assign_coords({new_dim: k})
                         .swap_dims({dim: coords}))

        # Record cycle slice with different coords
        new_dim_list.append(slice_new_dim)

    # Construct dataset along a new dimension based on new_dim
    data = xr.concat(new_dim_list, new_dim)

    # Merge dataset
    if non_dim_ds is not None:
        data = xr.merge([data, non_dim_ds])
    return data

def stack(ds, dims):
    
    attrs = get_attrs(ds)
    
    # String separator
    sep = ', '
    
    # Stack dim and drop nan
    stack_dim = sep.join(dims)
    ds = ds.stack({stack_dim: dims}).dropna(dim=stack_dim, how='all')
    ds2 = ds.copy(deep=True)
    
    # Generate new dimension labels
    lst = [sep.join(map(str, vs)) for vs in ds2.indexes[stack_dim].tolist()]
    
    # Add the new labels as a new variable
    ds[stack_dim] = xr.DataArray(lst, dims=stack_dim)
    
    # Copy original coordsin to new dataset
    coords = {dim: xr.DataArray(ds2[dim].values, dims=stack_dim)
              for dim in dims}
    ds = ds.assign_coords(coords)
        
    restore_attrs(ds, attrs)
    
    return ds

def process_stacked_groupby(ds, dim, func, *args):
    
    # Function to apply to stacked groupby
    def apply_fn(ds, dim, func, *args):
        
        # Get groupby dim
        groupby_dim = list(ds.dims)
        groupby_dim.remove(dim)
        groupby_var = ds[groupby_dim]
        
        # Unstack groupby dim
        ds2 = ds.unstack(groupby_dim).squeeze()#.dropna(dim)
        
        # perform function
        ds3 = func(ds2, *args)
            
#         try:
#             # perform function
#             ds3 = func(ds2, *args)
#         except:
#             ds3 = xr.Dataset()

        ds3 = (ds3
               .reset_coords(drop=True)
               .assign_coords(groupby_var)
               .expand_dims(groupby_dim)
             )
        return ds3
    
    # Get list of dimensions
    groupby_dims = list(ds.dims)
    
    # Get list of non-dimensional coordinates
    coords = list(set(ds.coords) - set(ds.dims))
    
    # Remove dimension not grouped
    groupby_dims.remove(dim)
    
    # Stack all but one dimensions
    stack_dim = '_'.join(groupby_dims)
    ds2 = ds.stack({stack_dim: groupby_dims})
    
    # Remove NaN elements
    # TODO any and all work for different situations
    ds2 = ds2.dropna(dim=stack_dim, how='all')
    
    # Groupby and apply
    ds2 = ds2.groupby(stack_dim, squeeze=False).map(apply_fn, args=(dim, func, *args))
    
    # Unstack
    ds2 = ds2.unstack(stack_dim)
    
    # Restore non-dim coords
    ds2 = (ds2
           .reset_coords(coords, drop=True)
           .assign_coords(ds[coords].reset_coords()))
    
    # Restore attrs
    for dim in groupby_dims:
        ds2[dim].attrs = ds[dim].attrs
    
    return ds2

# def process_stacked_groupby(ds, dim, func, *args, **kwargs):
    
#     # Function to apply to stacked groupby
#     def apply_fn(ds, dim, func, *args, dropna_how='all'):
        
#         # Get groupby dim
#         groupby_dim = list(ds.dims)
#         groupby_dim.remove(dim)
#         groupby_var = ds[groupby_dim]
        
#         # Unstack groupby dim
#         ds2 = ds.unstack(groupby_dim).squeeze()#.dropna(dim)
        
#         # perform function
#         ds3 = func(ds2, *args)
            
# #         try:
# #             # perform function
# #             ds3 = func(ds2, *args)
# #         except:
# #             ds3 = xr.Dataset()

#         ds3 = (ds3
#                .reset_coords(drop=True)
#                .assign_coords(groupby_var)
#                .expand_dims(groupby_dim)
#              )
#         return ds3
    
#     # Get list of dimensions
#     groupby_dims = list(ds.dims)
    
#     # Get list of non-dimensional coordinates
#     coords = list(set(ds.coords) - set(ds.dims))
    
#     # Remove dimension not grouped
#     groupby_dims.remove(dim)
    
#     # Stack all but one dimensions
#     stack_dim = '_'.join(groupby_dims)
#     ds2 = ds.stack({stack_dim: groupby_dims})
    
#     # Remove NaN elements
#     # TODO any and all work for different situations
#     ds2 = ds2.dropna(dim=stack_dim, how=dropna_how)
    
#     # Groupby and apply
#     ds2 = ds2.groupby(stack_dim, squeeze=False).map(apply_fn, args=(dim, func, *args), **kwargs)
    
#     # Unstack
#     ds2 = ds2.unstack(stack_dim)
    
#     # Restore non-dim coords
#     ds2 = (ds2
#            .reset_coords(coords, drop=True)
#            .assign_coords(ds[coords].reset_coords()))
    
#     # Restore attrs
#     for dim in groupby_dims:
#         ds2[dim].attrs = ds[dim].attrs
    
#     return ds2

# def groupby_all_except(ds, ex_dim, apply):
#     # Get list of dimensions
#     dims = list(ds.dims)
    
#     # Remove dimension not grouped
#     dims.remove(ex_dim)
    
#     # Stack all but one dimensions
#     groupby_dim = '_'.join(dims)
#     ds = ds.stack({groupby_dim: dims})
    
#     # Remove NaN elements
#     ds = ds.dropna(dim=groupby_dim)
    
#     # Groupby and apply
#     ds = ds.groupby(groupby_dim).map(apply)
    
#     # Unstack
#     ds = ds.unstack(groupby_dim)
#     return ds

# def stack_str(ds, dims):
#     # Create stacked dim name
#     stacked_dim = dims[0] + 'str'
    
#     # Stack dim and drop nan
#     ds = ds.stack({stacked_dim: dims}).dropna(dim=stacked_dim, how='all')
    
#     # Generate new dimension labels
#     lst = [' '.join(map(str, vs)) for vs in ds.indexes[stacked_dim].tolist()]
    
#     # Replace stacked dim with list
#     ds[stacked_dim] = lst
    
#     # Change stacked dim name to first dim name
#     ds = ds.rename({stacked_dim: dims[0]})
#     return ds

def process_stacked_groupby(ds, dim, func, *args):
    
    # Function to apply to stacked groupby
    def apply_fn(ds, dim, func, *args):
        
        # Get groupby dim
        groupby_dim = list(ds.dims)
        groupby_dim.remove(dim)
        groupby_var = ds[groupby_dim]
        
        # Unstack groupby dim
        ds2 = ds.unstack(groupby_dim).squeeze()#.dropna(dim)
        
        # perform function
        ds3 = func(ds2, *args)
            
#         try:
#             # perform function
#             ds3 = func(ds2, *args)
#         except:
#             ds3 = xr.Dataset()

        ds3 = (ds3
               .reset_coords(drop=True)
               .assign_coords(groupby_var)
               .expand_dims(groupby_dim)
             )
        return ds3
    
    # Get list of dimensions
    groupby_dims = list(ds.dims)
    
    # Get list of non-dimensional coordinates
    coords = list(set(ds.coords) - set(ds.dims))
    
    # Remove dimension not grouped
    groupby_dims.remove(dim)
    
    # Stack all but one dimensions
    stack_dim = '_'.join(groupby_dims)
    ds2 = ds.stack({stack_dim: groupby_dims})
    
    # Remove NaN elements
    # TODO any and all work for different situations
    ds2 = ds2.dropna(dim=stack_dim, how='all')
    
    # Groupby and apply
    ds2 = ds2.groupby(stack_dim, squeeze=False).map(apply_fn, args=(dim, func, *args))
    
    # Unstack
    ds2 = ds2.unstack(stack_dim)
    
    # Restore non-dim coords
    ds2 = (ds2
           .reset_coords(coords, drop=True)
           .assign_coords(ds[coords].reset_coords()))
    
    # Restore attrs
    for dim in groupby_dims:
        ds2[dim].attrs = ds[dim].attrs
    
    return ds2
