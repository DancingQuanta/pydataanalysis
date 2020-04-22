import os
import re
import numpy as np
from .xarray_utils import xr, reverse_sign_flip, pprint_label
from .statistics import unc_dim_to_vars
from .mpl import *
from matplotlib.backends.backend_pdf import PdfPages

import holoviews as hv
from holoviews import opts, dim
import hvplot.pandas
import hvplot.xarray

# Matplotlib
hv.extension('matplotlib')
hv.notebook_extension(display_formats=['html', 'svg'])
hv.output(fig='svg')

markers = hv.Cycle(['o', 's', '^', 'v', '*', 'D', 'h', 'x', '+', '8', 'p', '<', '>', 'd', 'H'])
fontsize = dict(labels=8, ticks=8, legend=8)
sublabels = dict(sublabel_size=14, sublabel_position=(-0.25, 0.95))
# sublabels = dict()
opts.defaults(
    opts.ErrorBars(elinewidth=1, show_legend=False),
    opts.Scatter(s=20, padding=0.1),
    opts.Layout(**sublabels))
#     opts.Curve(axiswise=True, framewise=True),
#     opts.Overlay(axiswise=True, framewise=True),
#     opts.NdOverlay(axiswise=True, framewise=True),
#     opts.NdLayout(axiswise=True, framewise=True),
#     fontsize=fontsize,

# default_legend_specs = hv.plotting.mpl.element.LegendPlot.legend_specs.copy()

def test_hook(plot, element):
    """
    Remove legend
    """
    print(plot.handles)
    print(element.data)

def sciticks(plot, element):
    """
    https://www.programcreek.com/python/example/102352/matplotlib.pyplot.ticklabel_format
    https://werthmuller.org/blog/2014/move-scientific-notation/
    http://greg-ashton.physics.monash.edu/setting-nice-axes-labels-in-matplotlib.html
    https://stackoverflow.com/questions/41752111/how-to-control-holoviews-y-tick-format
    https://peytondmurray.github.io/coding/fixing-matplotlibs-scientific-notation/#
    """
    ax = plot.handles['axis']
    
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

def remove_legend_title(plot, element):
    """
    Remove legend
    """
    ax = plot.handles['axis']
    l = ax.legend()
    l.set_title()

def show_grid(plot, element):
    """
    Remove legend
    """
    ax = plot.handles['axis']
    print(ax.get_axisbelow())
    print(ax.get_zorder())
    ax.set_zorder(-10)

def rotate_xticks(plot, element):
    """
    Rotate xtick
    """
    ax = plot.handles['axis']
    plt.setp(ax.get_xticklabels(), rotation=90)

def compute_marker(scatter):
    sel_data = scatter.data
    direction = sel_data['direction'].iloc[0]
    
    direction_markers = {
        'positive': 'o',
        'negative': '^'
    }

    return scatter.opts(marker=direction_markers[direction])


# Plots

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

def full_plot(ds, x, y):
    """
    Plot two subplots of two extreme currents and on each plot are two field directions
    """
    # Select data
    ds = (ds
          .pipe(reverse_sign_flip, y)
          .isel(reps=4)
          .sel(I=[ds['I'].min(), ds['I'].max()]))
    ds = ds[[x, y]].reset_coords(drop=True)
    
    plot = ds.hvplot(x=x, y=y, by='direction', groupby=['I'], kind='line', dynamic=False).layout('I')
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

def scatter_plot(ds, x, y, error_level=None, **kwargs):

    if error_level:
        ds = filter_uncertainty(ds, y, error_level)
        
    meas = ds.hvplot.scatter(x, y, dynamic=False, **kwargs)
    unc = ds.hvplot.errorbars(x, y, y + '_unc', dynamic=False, **kwargs)
    plot = unc * meas
    plot = (plot
            .opts(
                opts.Scatter(s=8, marker=markers, padding=0.1),
                opts.ErrorBars(elinewidth=1, show_legend=False)
            )
           )
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



class PlotPages():
    def __init__(self, dest_dir=None):
        self.dest_dir = dest_dir
        self.plots = []
        self.figs = []
        self.captions = []
        self.names = []
    
    def add(self, plot, name, caption):
        self.plots.append(plot)
        self.figs.append(hv.render(plot))
        self.names.append(name)
        self.captions.append(caption)
    
    def export(self):
        for i, plot in enumerate(self.plots):
            # Generate file path
            name = f'{i} {self.names[i]}.pdf'
            filepath = os.path.join(self.dest_dir, name)
            hv.save(plot, filepath)
    
    def multipage_export(self):
        filepath = os.path.join(self.dest_dir, 'multipage_pdf.pdf')
        with PdfPages(filepath) as pdf:
            for i, plot in enumerate(self.plots):
                fig = self.figs[i]
                pdf.savefig(fig)
