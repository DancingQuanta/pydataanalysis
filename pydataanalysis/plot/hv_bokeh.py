import os
import re
import numpy as np
from .xarray_utils import xr
# from .data import *
# from .statistics import *

import holoviews as hv
from holoviews import opts, dim
import hvplot.pandas
import hvplot.xarray

hv.extension('bokeh')

opts.defaults(
    opts.Scatter(width=400, size=3, padding=0.1,
                 axiswise=True, framewise=True),
    opts.ErrorBars(show_legend=False,
                   axiswise=True, framewise=True),
    opts.Curve(axiswise=True, framewise=True),
    opts.Overlay(width=400, padding=0.1,
                 axiswise=True, framewise=True),
    opts.NdOverlay(width=400, padding=0.1,
                 axiswise=True, framewise=True),
    opts.NdLayout(axiswise=True, framewise=True))

def compute_marker(scatter):
    sel_data = scatter.data
    direction = sel_data['direction'].iloc[0]
    direction_markers = {
        'positive': 'circle',
        'negative': 'triangle'
    }
    return scatter.opts(marker=direction_markers[direction])
    

def full_plot(ds, x, y):
    """
    Plot two subplots of two extreme currents and on each plot are two field directions
    """
    # Select data
    ds = (ds
          .isel(reps=4)
          .sel(I=[ds['I'].min(), ds['I'].max()]))
    ds = ds[[x, y]].reset_coords(drop=True)
    
    plot = ds.hvplot.scatter(x='Bext', y='y1', by='direction', groupby=['I'], dynamic=False).layout('I')
    plot = plot.opts(opts.NdOverlay(aspect=1.6, legend_position='inner'), opts.NdLayout(tight=True))
    return plot

def grid_reps_direction_current_plot(ds, x, y):
    grid = (ds.pipe(reverse_sign_flip, y)
            .hvplot.line(x=x, y=y, by='reps', row='direction', col='I', groupby=[], dynamic=False)
            .opts(opts.NdOverlay(show_legend=False))
           )
    return grid

def grid_direction_reps_current_plot(ds, x, y):
    grid = (ds.pipe(reverse_sign_flip, y)
            .hvplot.line(x=x, y=y, by='direction', row='reps', col='I', groupby=[], dynamic=False)
            .opts(opts.NdOverlay(show_legend=False))
           )
    
    bottom_lgd_specs = dict(ncol=2, loc='lower center',
                            bbox_to_anchor=(0.5, -1.3, 1., .102),
    #                         bbox_to_anchor=(0, -0.25),
                            borderaxespad=0.1)
    hv.plotting.mpl.element.LegendPlot.legend_specs['bottom'] = bottom_lgd_specs
    legend_opts = {'NdOverlay': dict(show_legend=True, legend_position='bottom')}
    
    ncols = len(ds['reps'])
    half_cols = ncols // 2
    
    grid_index = (ds['reps'].values[half_cols-1], ds['I'].values[-1])
    grid[grid_index] = grid[grid_index].opts(legend_opts)
    return grid

#     fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False, sharex=True)
    
#     xlabel = pprint_label(ds[x])
#     ylabel = pprint_label(ds[y])

#     for i, ax in enumerate(axes):
#         sub = ds.isel(I=i)
#         current = sub['I']
#         sub = sub.reset_coords(drop=True)
#         pos = sub.sel(direction='positive')#.to_dataframe()
#         neg = sub.sel(direction='negative')#.to_dataframe()
#         ax.plot(pos[x], pos[y], label='Ascending Field')
#         ax.plot(neg[x], neg[y], label='Descending Field')

#         if i == 0:
#             ax.set_ylabel(ylabel)
#             ax.legend(loc='lower center')
            
#         ax.set_xlabel(xlabel)
#         ax.set_title(pprint_title(current))
    
#     fig.tight_layout()
    
def plot_data(data, x, y, dim=None, kind='scatter', **kwargs):
    data = data.copy(deep=True)
    if hasattr(data, 'to_dataset'):
        data = data.to_dataset()
   
    if 'unc' in data[y].dims:
        data = data.pipe(unc_dim_to_vars)
    
    # we are using bidx for scatter, what dimensions are left?
    groupby = list(data[y].dims)
    if len(data[x].dims) == 1:
        groupby.remove(data[x].dims[0])
    elif dim is not None:
        groupby.remove(dim)
    else:
        raise ValueRrror('specify dimension to plot along')
        
#     data[y] = data[y].pipe(xmagnitude_thousand)
#     data[y].attrs['long_name'] = y
    
    labels = {'xlabel': pprint_label(data[x]),
              'ylabel': pprint_label(data[y])}
    
    kwargs.update(labels)
    kwargs_err = {}
    kwargs_err.update(labels)
    
    plot = (data
            .to_dataframe()
            .hvplot(x=x, y=y, groupby=groupby, kind=kind, dynamic=False, **kwargs))
#             .apply(compute_marker))
    
    y_err = y + '_unc'
    if y + '_unc' in data:
        error = (data
                 .to_dataframe()
                 .hvplot.errorbars(x=x, y=y, yerr1=y_err, groupby=groupby, **kwargs_err, dynamic=False))
        plot = error * plot
        
    return plot

def scatter_plot(ds, x, y, error_level=None):
    

    if error_level:
        ds = filter_uncertainty(ds, y, error_level)
        
    meas = ds.hvplot.scatter(x, y, dynamic=False)
    unc = ds.hvplot.errorbars(x, y, y + '_unc', dynamic=False)
    plot = unc * meas
    plot = (plot
            .opts(
                opts.Scatter(s=8, marker=markers, padding=0.1),
                opts.ErrorBars(elinewidth=1, show_legend=False)
            )
           )
    return plot

def peak_width_data(ds, x, y):
    """
    Transfer information from the original dataset to the peak width.
    """
    
    # X coordinates
    width_x = ds[['left', 'right']].to_array(dim='line', name=ds[x].name)
    width_x.attrs = ds[x].attrs
    
    # Y coordinates
    width_y = ds['width_h'].rename(ds[y].name)
    width_y.attrs = ds[y].attrs
    
    # Combine
    ds = xr.merge([width_x, width_y.broadcast_like(width_x)])
    return ds

def plot_widths(ds, x, y):
    
    ds = peak_width_data(ds, x, y)

    # Groupby dimensions other than 'line'
    remaining_dims = list(ds.dims)
    remaining_dims.remove('line')
    
#     ds['y'] = ds['y'].pipe(xmagnitude_thousand)
    
    # Plot a line
    plot = (ds.to_dataframe()
            .hvplot.line(x=x, y=y,
                         groupby=remaining_dims,
                         label='width',
                         dynamic=False))
    return plot

def layout_direction(plot):

    layout = plot.layout('direction')

    # Positive
    positive = layout['positive'].opts(
        show_legend=False,
        title='Ascending field'
    )

    # Negative
    negative = layout['negative'].opts(
        show_legend=True, 
#         ylabel='',
#         yaxis=None,
        title='Descending field'
    )
    
    plot = positive + negative
    plot = (plot.opts(sublabel_format=None, tight=True)
            .opts(opts.Scatter(aspect=1.6)))
    return plot

def overlay_direction(plot):

    # Positive
    positive = plot.select(direction='positive').opts(
#         label='Ascending field'
    )

    # Negative
    negative = plot.select(direction='negative').opts(
#         label='Descending field'
    )
    
    plot = positive * negative
    print(plot)
    plot = plot.opts(aspect=1.6)
    return plot

def plot_layout_direction(ds, x, y, dim):

    plot = plot_data(data3, x, y, dim).layout('direction')

    plot = plot.opts(
        opts.Scatter(axiswise=True, framewise=True),
        opts.Curve(axiswise=True, framewise=True),
        opts.Overlay(axiswise=True, framewise=True),
        opts.NdLayout(axiswise=True, framewise=True))
    return plot

def error_spread(data, x, y, y_err, dim=None, **kwargs):
    
    if hasattr(data, 'to_dataset'):
        data = data.to_dataset()
   
    # we are using bidx for scatter, what dimensions are left?
    groupby = list(data[y].dims)
    if len(data[x].dims) == 1 and dim is not None:
        raise
        
    if len(data[x].dims) == 1:
        groupby.remove(data[x].dims[0])
        
    if dim:
        groupby.remove(dim)
    
    ymin = data[y] - data[y_err]
    ymax = data[y] + data[y_err]
    
    data = data.assign({y+' min': ymin,
                        y+' max': ymax})
    
    spread = (data.to_dataframe()
              .hvplot.area(x=x, y=y+' min', y2=y+' max', groupby=groupby, dynamic=False, alpha=0.5, **kwargs))
    
    return spread

def error_plot(data, x, y, dim=None):
    
    # Scatter plot
    scatter = plot_data(data, x, y, dim, label='data', sort_date=False).opts(color='blue')
    
    # Ensure that statistics are broadcast along reduced dimension for vis
    data = data.broadcast_like(data[dim])
    
    # Plots
    mean_line = plot_data(data, x, 'mean', dim, kind='line', color='indianred', label='mean', sort_date=False)
    mean_spread = error_spread(data, x, 'mean', 'std', color='indianred', label='std', sort_date=False)
    
    median_line = plot_data(data, x, 'median', dim, kind='line', color='limegreen', label='median', sort_date=False)
    median_spread = error_spread(data, x, 'median', 'mad', color='limegreen', label='mad', sort_date=False)
    
    plot = mean_spread * median_spread * mean_line * median_line * scatter
    plot = plot.opts(ylabel=y)
    
    return plot

def peak_param_error_plot(ds, x, param, layout):
    ds = sel_rename(ds, 'params', 'data', param)

    plot = error_plot(ds, x, param, x)
    plot = plot.layout(layout).opts(
        opts.Area(axiswise=False, framewise=False),
        opts.Curve(axiswise=False, framewise=False),
        opts.Scatter(axiswise=False, framewise=False),
        opts.Overlay(axiswise=False, framewise=False, legend_position='best', aspect=1.6),
        opts.NdLayout(axiswise=False, framewise=False, tight=True, fig_size=90),
    ).cols(3)
    return plot

def peak_params_plot(ds, x):
    plot_x = (plot_data(ds, x, 'center')
              .overlay('direction'))
    plot_y = (plot_data(ds, x, 'height')
              .overlay('direction'))
    plot_width = (plot_data(ds, x, 'fwhm')
                  .overlay('direction'))
    return plot_x + plot_y + plot_width

def mpl_line_overlay_grid(ds, x, y, overlay, row, col):
    # Number of rows and cols
    nrows = len(ds[row])
    ncols = len(ds[col])
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharey='row', sharex='col')
    
    xlabel = pprint_label(ds[x])
    print(xlabel)
    ylabel = pprint_label(ds[y])
    print(ylabel)
    row_label = pprint_label(ds[row])
    print(row_label)
    col_label = pprint_label(ds[col])
    print(col_label)
    
    for i, r in enumerate(ds[row]):
        for j, c in enumerate(ds[col]):
#             print(i, j)
#             print(r, c)
            ax = axes[i, j]
            sub = ds.isel({row:i, col:j})
            ax.plot(sub[x], sub[y])
            sub.plot.scatter(x=x, y=y, hue=overlay, hue_style='discrete', ax=ax)
            
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            
            # Top
            if i == 0:
                col_value = pprint_value(sub[col])
                ax.set_xlabel(col_value)
                ax.xaxis.set_label_position('top')
            
            # Bottom
            if i == len(ds[row])-1:
                ax.set_xlabel(xlabel)
                
            # Left
            if j == 0:
                ax.set_ylabel(ylabel)

            # Right
            if j == len(ds[col])-1:
                row_value = pprint_value(sub[row])
                ax.set_ylabel(row_value, rotation=270, labelpad=15)
                ax.yaxis.set_label_position('right')
    
    fig.tight_layout()
    return fig

def mpl_error_plot(ds, x, y, ax):
    
    # Plots
    ds.line(x, 'mean', ax=ax, color='indianred', label='mean')
    mpl_error_spread(data, x, 'mean', 'std', color='indianred', alpha=0.5, label='std')
    
    ds.line(x, 'median', ax=ax, color='indianred', label='median')
    mpl_error_spread(data, x, 'median', 'mad', color='indianred', alpha=0.5, label='mad')
    
    # Scatter plot
    ds.scatter(x, y, ax=ax, color='blue', label='data')
    
    
def mpl_error_spread(ds, x, y, y_err, ax, **kwargs):
    
    ymin = data[y] - data[y_err]
    ymax = data[y] + data[y_err]
    
    data = data.assign({y+' min': ymin,
                        y+' max': ymax})
    
    ax.fill_between(data[x], data[y+' min'], data[y+' max'], **kwargs)
    
def mpl_peak_param_error_plot(ds, x, y, row, col):

    # Ensure that statistics are broadcast along reduced dimension for vis
    ds = ds.broadcast_like(ds[x])
    
    # Number of rows and cols
    nrows = len(ds[row])
    ncols = len(ds[col])
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4), sharey=True, sharex=True)
    
    print(axes)
    
#     def plot_sweep(ax, data, variable, sweep):
#         ax.set_title(f"Sweep: {sweep}")
#         ax.set_xlabel(label_format.format(**data['Field'].attrs))
#         ax.grid()

#         # Select data
#         line = (data.sel(sweep=sweep)
#                 .dropna(dim='Nominal_Field'))

#         # Plot
#         plt_kwargs = {
#             'marker': 'o',
#             'markersize': 2,
#             'linestyle': '-',
#             'linewidth': 1,
#         }
#         ax.plot(line['Nominal_Field'],
#                 line[variable],
#                 **plt_kwargs,
#                 label='Nominal')
        
#         ax.plot(line['Field'],
#                 line[variable],
#                 **plt_kwargs,
#                 label='Flux-corrected')
        
#     # Initial
#     ax1 = fig.add_subplot(gs[0, 2:6])
#     plot_sweep(ax1, data, variable, 'initial')
#     ax1.set_ylabel(label_format.format(**data[variable].attrs))
# #     ylims = ax1.get_ylim()
# #     y_range = ylims[1] - ylims[0]
# #     y_steps = y_range / len(ax1.get_ydata())
#     ax1.set_ylim(0, ax1.get_ylim()[1])
    
#     # Down
#     ax2 = fig.add_subplot(gs[1, :4])
#     plot_sweep(ax2, data, variable, 'down')
#     ax2.set_ylabel(label_format.format(**data[variable].attrs))
    
#     # Up
#     ax3 = fig.add_subplot(gs[1, 4:])
#     plot_sweep(ax3, data, variable, 'up')

#     # Make down and up the same limits
#     equalize_axes([ax2, ax3])
# #     ylim2 = ax2.get_ylim()
# #     ylim3 = ax3.get_ylim()
# #     yb = min([ylim2[0], ylim3[0]])
# #     yt = max([ylim2[1], ylim3[1]])
# #     ax2.set_ylim((yb, yt))
# #     ax3.set_ylim((yb, yt))
    
# #     xlim2 = ax2.get_xlim()
# #     xlim3 = ax3.get_xlim()
# #     xb = min([xlim2[0], xlim3[0]])
# #     xt = max([xlim2[1], xlim3[1]])
# #     ax2.set_xlim((xb, xt))
# #     ax3.set_xlim((xb, xt))
    
# #     leg = ax1.legend()
#     setup_legend(ax1.legend())
#     fig.tight_layout()
#     fig.savefig(os.path.join(fig_dir, f'Flux-corrected {variable}.png'))

# def arrow_direction(x, y, direction):
#     if direction == 'positive':
#         arrow_mark = '>'
#     else:
#         arrow_mark = '<'
        
#     return hv.Arrow(x, y, 'Direction', arrow_mark)

# def plot_scan(Field):
#     sel_data = data.sel(Field=Field)
#     x = 'Position'
#     y = 'sig1'
    
#     scatter = sel_data.to_dataframe().hvplot.scatter(x=x, y=y)
    
#     direction = sel_data['direction'].iloc[0]
#     x = np.max(sel_data['Position']).item()*0.5
#     y = np.max(sel_data['sig1']).item()*0.5
#     arrow = arrow_direction(x, y, direction)
#     return scatter * arrow

# dmap = (hv.DynamicMap(plot_scan, kdims=['Field'])
#         .redim.values(Field=list(data.Field.values)))
# dmap

# scatter = data.to_dataframe().hvplot.scatter(x='Position', y='sig1', groupby=['B'])
# def compute_arrow(scatter):
#     print(scatter.data)
#     sel_data = scatter.data
#     if sel_data['direction'].iloc[0] == 'positive':
#         arrow_mark = '>'
#     else:
#         arrow_mark = '<'
        
#     x = np.max(sel_data['Position']).item()*0.5
#     y = np.max(sel_data['sig1']).item()*0.5
#     return hv.Arrow(x, y, 'Direction', arrow_mark)

# scatter * scatter.apply(compute_arrow)

# grouped = hv.Dataset(data).groupby('Field', dynamic=True)

# def compute_arrow(ds):
#     sel_data = ds.data
#     if sel_data['direction'] == 'positive':
#         arrow_mark = '>'
#     else:
#         arrow_mark = '<'
#     return hv.Arrow(np.max(sel_data['Position']).item()*0.5, np.max(sel_data['Signal']).item()*0.5, 'Direction', arrow_mark)

# def to_scatter(ds):
#     return ds.dframe().hvplot.scatter(x='Position', y='Signal')

# grouped.apply(to_scatter) * grouped.apply(compute_arrow)

# directions = ['positive', 'negative']
# markers = ['circle', 'triangle']
#                     marker=dim('direction').bin(directions, markers)),
