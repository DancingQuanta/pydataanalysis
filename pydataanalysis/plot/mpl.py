import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

from pydataanalysis.xarray_utils import reverse_sign_flip, pprint_label, pprint_label_value
from pydataanalysis.statistics import unc_dim_to_vars

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')

# from matplotlib.ticker import ScalarFormatter
# yfmt = ScalarFormatter()
# yfmt.set_powerlimits((0,0))
# yfmt.set_scientific(True)

# import matplotlib.style as style
# style.use('seaborn-bright')

# options = {'Scatter': {
#     'marker': markers,
#     'color': hv.Cycle('tab10')
# }

# def latexify(fig_width=None, fig_height=None, columns=1):
#    """Set up matplotlib's RC params for LaTeX plotting.
#    Call this before plotting a figure.

#    Parameters
#    ----------
#    fig_width : float, optional, inches
#    fig_height : float,  optional, inche
#    columns : {1, 2}
#    """

#    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

#    # Width and max height in inches for IEEE journals taken from
#    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

#    assert(columns in [1,2])

#    if fig_width is None:
#        fig_width = 3.39 if columns==1 else 6.9 # width in inches

#    if fig_height is None:
#        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
#        fig_height = fig_width*golden_mean # height in inches

#    MAX_HEIGHT_INCHES = 8.0
#    if fig_height > MAX_HEIGHT_INCHES:
#        print("WARNING: fig_height too large:" + fig_height +
#              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
#        fig_height = MAX_HEIGHT_INCHES

params = {
    'text.latex.preamble': [r'\usepackage{gensymb}'],
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8, # was 10
    'legend.fontsize': 8, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': True,
    #'figure.figsize': [fig_width, fig_height],
    'font.family': 'serif',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'errorbar.capsize': 2,
    'axes.axisbelow': True,
}

matplotlib.rcParams.update(params)

# latexify()

# def sublabel(label, ax, fmt=None, size=18, pos=(-0.35, 0.85)):
#     from matplotlib.offsetbox import AnchoredText
#     at = AnchoredText(self.sublabel_format.format(**labels), loc=3,
#                   bbox_to_anchor=pos, frameon=False,
#                   prop=dict(size=self.sublabel_size, weight='bold'),
#                   bbox_transform=axis.transAxes)
#     at.patch.set_visible(False)
#     axis.add_artist(at)

def setup_legend(leg):
    frame = leg.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('0.0')
    frame.set_linewidth('1.0')
    leg.set_zorder(10e6)

def hide_spines(ax):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
# def grid_direction_reps_current_plot(ds, x, y):
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

def mpl_overlay_grid(ds, x, y, overlay, row, col, figsize=None, **kwargs):
    
    ds = ds.pipe(reverse_sign_flip, y)
    
    # Number of rows and cols
    nrows = len(ds[row])
    ncols = len(ds[col])
    
    fig, axes = plt.subplots(nrows, ncols, sharey='row', sharex='col', figsize=figsize)
    
    xlabel = pprint_label(ds[x])
    ylabel = pprint_label(ds[y])
    
    for i, r in enumerate(ds[row]):
        for j, c in enumerate(ds[col]):
            ax = axes[i, j]
            sub = ds.isel({row:i, col:j})
#             sub.plot.scatter(x=x, y=y, hue=overlay, hue_style='discrete', ax=ax, **kwargs)

            df = sub.reset_coords().to_dataframe()
            for k, grp in df.groupby(overlay):
                ax.plot(grp[x], grp[y], label=str(k))
    
#             ax.set_xlabel(None)
#             ax.set_ylabel(None)
            
            if i == 0 and j == 0:
                setup_legend(ax.legend())
#             else:
#                 ax.get_legend().remove()
            
            # Top
            if i == 0:
                col_value = pprint_label_value(sub[col])
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
                row_value = pprint_label_value(sub[row])
                ax.set_ylabel(row_value, rotation=270, labelpad=15)
                ax.yaxis.set_label_position('right')
            
            hide_spines(ax)
    
    fig.tight_layout()

def mpl_error_spread(x, y, y_err, ax, **kwargs):
    
    lower = y - y_err
    upper = y + y_err
    
    ax.fill_between(x, lower, upper, **kwargs)

def mpl_error_line_plot(ax, x, y, y_err, label):
    
    # Line plot
    ax.plot(x, y, ls='-', color='indianred', label=label)
    
    # Area plot
    mpl_error_spread(x, y, y_err, ax, color='indianred', alpha=0.5, label=f"{label} band")
    
def mpl_scatter_error_plot(ax, x, y, y_central, y_dispersion):
    
    # Scatter plot
    ax.plot(x, y, ls='None', marker='.', color='blue', label='data')
    
    # Line plot
#     ds.plot.line(x, 'mean', ax=ax, color='indianred', label='mean')
    ax.plot(x, y_central, ls='-', marker='None', color='indianred', label='mean')
    
    # Area plot
    mpl_error_spread(x, y_central, y_dispersion, ax, color='indianred', alpha=0.5, label='std')
    
def mpl_error_plot_df(df, ax):
    
    # Scatter plot
    df['data'].plot(ax=ax, linestyle='None', marker='.', color='blue', label='data')
    
    # Line plot
#     ds.plot.line(x, 'mean', ax=ax, color='indianred', label='mean')
    df['mean'].plot(ax=ax, linestyle='-', marker='None', color='indianred', label='mean')
    
    # Area plot
    mpl_error_spread(df.index, df['mean'], df['std'], ax, color='indianred', alpha=0.5, label='std')
    
#     ds.line(x, 'median', ax=ax, color='indianred', label='median')
#     mpl_error_spread(ds, x, 'median', 'mad', color='indianred', alpha=0.5, label='mad')
    

def mpl_error_grid(ds, dim, row, col, figsize=None, share_all=False, **kwargs):
    # Variables
    raw =  'data'
    central = 'mean'
    dispersion = 'std'
    
    # Broadcast variables
    ds = ds[[raw, central, dispersion]].broadcast_like(ds[raw])
    
    # Number of rows and cols
    nrows = len(ds[row])
    ncols = len(ds[col])
    
    sharey = True if share_all else 'row'
    sharex = True if share_all else 'col'
    
    fig, axes = plt.subplots(nrows, ncols, sharey=sharey, sharex=sharex, figsize=figsize)
    
    xlabel = pprint_label(ds[dim])
    ylabel = pprint_label(ds[raw])
    
    for i, r in enumerate(ds[row]):
        for j, c in enumerate(ds[col]):
            ax = axes[i, j]
            sub = ds.isel({row:i, col:j})
            mpl_scatter_error_plot(ax, sub[dim], sub[raw], sub[central], sub[dispersion])
#             df = sub.reset_coords(drop=True).to_dataframe()
#             mpl_error_plot_df(df, ax)

#             ax.set_xlabel(None)
#             ax.set_ylabel(None)
            
            if i == 0 and j == 0:
#                 handles, labels = ax.get_legend_handles_labels()
#                 leg = ax.legend(handles[::-1], labels[::-1])
                leg = ax.legend()
                setup_legend(leg)
#             else:
#                 ax.get_legend().remove()
            
            # Top
            if i == 0:
                col_value = pprint_label_value(sub[col])
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
                row_value = pprint_label_value(sub[row])
                ax.set_ylabel(row_value, rotation=270, labelpad=15)
                ax.yaxis.set_label_position('right')
    
            hide_spines(ax)
        
    fig.tight_layout()
    
def mpl_scatter_error_overlay(da, dim, overlay, ax):
    df = da.reset_coords(drop=True).to_dataset(dim='unc').to_dataframe()
    df = df.unstack(overlay)
    
    df.plot(ax=ax, y='Measurand', yerr='Uncertainty', fmt='.')
    
    leg = ax.legend()
    setup_legend(leg)
    leg.set_title(overlay)
    
    xlabel = pprint_label(da[dim])
    ylabel = pprint_label(da)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    hide_spines(ax)
    
#     for k, grp in df.groupby(level=overlay):
# #         ax.errorbar(grp.index, gra['Measurand'], yerr=grp['Uncertainty'], label=str(k), fmt='.')
#         grp.plot(ax=ax, y='Measurand', yerr='Uncertainty', label=str(k), fmt='.')

    
def direction_fit_plot(ds, title=None):
    
    # Setup figure
    fig, ax = plt.subplots()
    
    # Extract variables
    xdata = data['I']
    ydata = data['data']
    y_fitdata = data['best_fit']
    
    # Plot data
    if 'unc' in ydata.dims:
        # Error bar
        ydata = ydata.pipe(unc_dim_to_vars)
        yunc = ydata[f"{y}_unc"]
        ydata = ydata[y]
        ax1.errorbar(xdata, ydata, yerr=yunc, fmt='o', ms=2, label='data')
    else:
        ax1.plot(xdata, ydata, 'o', ms=2, lw=0, label='data')
    
    # Plot best fit
    if 'unc' in y_fitdata.dims:
        y_fitdata = y_fitdata.pipe(unc_dim_to_vars)
        yunc = y_fitdata[f"{y_fit}_unc"]
        y_fitdata = y_fitdata[y_fit]
        mpl_error_line_plot(ax1, xdata, y_fitdata, yunc, 'fit')
    else:
        ax1.plot(xdata, y_fitdata, '-', lw=1, label='fit')
    
    # Plot residuals
    y_res = ydata - y_fitdata
    ax2.plot(xdata, y_res, 'o', ms=2, lw=0)

    # Adjust residual plot
    min_val = np.abs(np.min(y_res))
    max_val = np.max(y_res)
    if min_val > max_val:
        max_val = min_val*1.1
        min_val = -min_val*1.1
    else:
        max_val = max_val*1.1
        min_val = -max_val*1.1
    ax2.set_ylim(max_val, min_val)
    ax2.axhline(0, ls='--', color='k', lw=0.5)

    if title is not None:
        ax1.set_title(title)
    
    # Labels
    ax1.set_ylabel(pprint_label(ydata))
    ax2.set_ylabel('Residual')
    ax2.set_xlabel(pprint_label(xdata))
    
    # Legend
    leg = ax1.legend()
    setup_legend(leg)
    
    # Spines
    hide_spines(ax1)
    hide_spines(ax2)

    fig.subplots_adjust(hspace=0)
    ax2.set_zorder(-1)
