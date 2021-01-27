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

def fit_plot(data, x, y, y_fit, title=None):
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Extract variables
    xdata = data[x]
    ydata = data[y]
    y_fitdata = data[y_fit]
    
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

### xarray ###

def xgplot(fn, ds, x, y, hue=None, ncol=1, fn_kwargs={}, **kwargs):

    # Create FacetGrid
    g = xr.plot.FacetGrid(ds, **kwargs)

    # Apply plot function to each axis
    g.map_dataset(fn, x, y, hue, 'discrete', **fn_kwargs)

    # Set ylabels
    g.set_ylabels(xr.plot.utils.label_from_attrs(ds[y]))

    return g

def xgerror_plot(fn, da, x, hue=None, ncol=1, fn_kwargs={}, **kwargs):

    ds = da.reset_coords(drop=True)

    # Split uncertainty tinto individual variables

    if 'unc' in ds:
        ds = ds.to_dataset(dim='unc')

    # Create FacetGrid
    g = xr.plot.FacetGrid(ds, **kwargs)

    # Apply plot function to each axis
    g.map_dataset(fn, x, 'Measurand', hue, 'discrete', **fn_kwargs)

    # Set ylabels
    g.set_ylabels(xr.plot.utils.label_from_attrs(da))

def scatter_error_plot(ax=None, ds=None, x=None, y=None, xerr=None, yerr=None,
                       hue=None, fmt='.', marker=None, linestyle=None, **kwargs):

    if 'unc' in ds[y].dims:
        ds = ds[y].to_dataset(dim='unc')
        y = 'Measurand'
        yerr = 'Uncertainty'

    variables = [x, y]

    if yerr and yerr in ds:
        variables.append(yerr)

    if xerr and xerr in ds:
        variables.append(xerr)

    kwargs = {}

    if yerr or xerr:
        kwargs['fmt'] = fmt
    else:
        kwargs['marker'] = marker
        kwargs['linestyle'] = linestyle

    ds = ds[variables]

    if hue:

        for j, h in enumerate(ds[hue]):
            df = (ds.sel({hue:h}, drop=True)
                  .to_dataframe().reset_index()
                  .dropna().sort_values(x))

            # Observations
            df.plot(ax=ax, x=x, y=y, xerr=xerr, yerr=yerr,
                    legend=False, **kwargs)
    else:
        (ds.to_dataframe().reset_index()
         .sort_index().dropna()
         .plot(ax=ax, x=x, y=y, xerr=xerr, yerr=yerr,
               legend=False, **kwargs))

def reg_error_plot(ax=None, ds=None, x=None, y=None, hue=None, fmt='', **kwargs):

    # Setup uncertainty if available
    yerr = f"{y}_unc"

    variables = [x, y, 'pred', 'ci_lower', 'ci_upper']

    if yerr in ds:
        fmt = '.'
        variables.append(yerr)
    else:
        yerr = None
        fmt = None

    ds = ds[variables]

    if hue:

        for i, h in enumerate(ds[hue]):
            sub = ds.sel({hue:h}, drop=True)

            color = next(ax._get_lines.prop_cycler)['color']

            df = (sub.to_dataframe().reset_index()
                  .sort_values(x).dropna())

            # Observations
            df.plot(ax=ax, x=x, y=y, yerr=yerr,
                    fmt=fmt, color=color, label=f'obs, {i}', legend=False)

            # Prediction
            df.plot(ax=ax, x=x, y='pred',
                    color=color, label=f'pred, {i}', legend=False)

            # Confidence interval
            ax.fill_between(df[x], df['ci_lower'], df['ci_upper'],
                            color=color, alpha=0.4, label=f'ci, {i}')
    else:

        df = (ds.to_dataframe().reset_index()
              .sort_values(x).dropna())

        # Observations
        df.plot(ax=ax, x=x, y=y, yerr=yerr,
                fmt=fmt, label='obs', legend=False)

        # Prediction
        df.plot(ax=ax, x=x, y='pred', label='pred', legend=False)

        # Confidence interval
        ax.fill_between(df[x], df['ci_lower'], df['ci_upper'],
                        alpha=0.4, label='ci')

def add_xfacetgrid_tablelegend(self, col_labels, **kwargs):

    # Create custom table legend
    figlegend = tablelegend(
        self.axes.flat[-1], ncol=3,
        col_labels=col_labels,
        row_labels=list(self._hue_var.values),
        title_label=self._hue_label,
        bbox_transform=self.fig.transFigure,
        loc="center right",
        # loc='lower center',
        **kwargs,
    )
    self.figlegend = figlegend

    # Add new legend
    self.fig.legends.append(figlegend)
    figlegend._remove_method = self.fig.legends.remove
    self.fig.stale = True

    # Draw the plot to set the bounding boxes correctly
    self.fig.draw(self.fig.canvas.get_renderer())

    # Calculate and set the new width of the figure so the legend fits
    legend_width = figlegend.get_window_extent().width / self.fig.dpi
    figure_width = self.fig.get_figwidth()
    self.fig.set_figwidth(figure_width + legend_width)

    # Draw the plot again to get the new transformations
    self.fig.draw(self.fig.canvas.get_renderer())

    # Now calculate how much space we need on the right side
    legend_width = figlegend.get_window_extent().width / self.fig.dpi
    space_needed = legend_width / (figure_width + legend_width) + 0.02
    # margin = .01
    # _space_needed = margin + space_needed
    right = 1 - space_needed

    # Place the subplot axes to give space for the legend
    self.fig.subplots_adjust(right=right)
