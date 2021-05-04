import numpy as np
import xarray as xr
import statsmodels.api as sm
from scipy.stats import sem, linregress
from scipy import odr
from scipy import optimize
from lmfit.models import GaussianModel, RectangleModel, LorentzianModel
import pybroom as br

from .xarray_utils import *
from .statistics import *

def np_linear_reg(x, y):
    
#     x, y = dropna(x, y)
    
    # Fit data
    p, C_p = np.polyfit(x, y, 1, cov=True)

    # Results
    params = np.stack([p, np.sqrt(np.diag(C_p))])
    # np.polyfit params order gradient, intercept, reverse it
    # to intercept, gradient
    params = np.flip(params, axis=1)
    intercept, gradient = params[:,0], params[:,1]

    return gradient, intercept

def sm_linear_reg(x, y):
    
    x, y = dropna(x, y)
    
    # Fit data
    mod = sm.OLS(y, sm.add_constant(x))

    # Results
    res = mod.fit()
    params = np.stack([res.params, res.bse])
    intercept, gradient = params[:,0], params[:,1]

    return gradient, intercept, res.rsquared

def xsmregress(ds, x, y, dim):
    ds = ds.reset_coords()
    
#     results = xr.apply_ufunc(
#         sm_linear_reg, ds[x], ds[y],
#         input_core_dims=[[dim], [dim]],
#         output_core_dims=[['unc', 'params']],
#         vectorize=True
#     )
    
#     param_names = ['intercept', 'gradient'] # TODO check this order!
#     unc_names = ['Measurand', 'Uncertainty']
#     results = (results
#                .assign_coords(params=param_names, unc=unc_names)
#                .rename('fit_results')
#               )
    
    results = xr.apply_ufunc(
        sm_linear_reg, ds[x], ds[y],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[['unc'], ['unc'], []],
        vectorize=True
    )
    
    var_names = ['gradient', 'intercept', 'r_squared']
    unc_names = ['Measurand', 'Uncertainty']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}
    results = (xr.Dataset(data_vars)
               .assign_coords(unc=unc_names))

    return results

def calc_r_squared(y, yhat, p):
    """
    Calculate rsquared and adjusted rsquared score

    Args:
        y (array): Vector representing dependent variable
        yhat (array): Vector of predicted values
        p (int): Number of explanatory variables

    Return:
        float: rsquared
        float: adjusted rsquared
    """
    e = y - yhat
    ybar = np.mean(y)

    sst = np.sum((y-ybar)**2)
    sse = np.sum((yhat-ybar)**2)
    ssr = np.sum((y-yhat)**2)

    r_squared = 1 - (ssr / sst)
    r_squared = sse / sst

    # Number of observations
    n = y.shape

    # Adjusted rsquared
    adj_rsquared = 1 - (1-rsquared**2)*(float(n-1)/float(n-p-1))

    return r_squared, adj_rsquared

def calc_se(y, yhat):
    """
    Calculate Standard error of estimate

    Args:
        y (array): Vector representing dependent variable
        yhat (array): Vector of predicted values

    Returns:
        float: Standard error of estimate
    """
    yhat = f(p, x)
    e = y - yhat
    se = np.sqrt(np.sum(e)/np.count(e))

    return se

def orthoregress(x, y, sx, sy):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.
    
    Arguments:
        x: x data
        y: y data
    Returns:
        [m, c, nan, nan, nan]
    """
    def f(p, x):
        '''Linear function y = a + b*x '''
        # p is a vector of the parameters.
        # x is an array of the current x values.
        # y is in the same format as the x passed to Data or RealData.
        #
        # Return an array in the same format as y passed to Data or RealData.

        return p[0] * x + p[1]
    
    # Initial estimation to be passed as guess to odr
    linreg = linregress(x, y)

    if np.isnan(linreg[0]):
        print('Nan')
        print(x)
        print(y)
    
    # ODR fit
    mod = odr.Model(f)
    dat = odr.RealData(x, y, sx=sx, sy=sy)
    od = odr.ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()
    
    params = np.stack([out.beta, out.sd_beta])
    gradient, intercept = params[:,0], params[:,1]
    
#     residuals = y - f(out.beta, x)
#     ss_res = np.sum(residuals**2)
#     ss_tot = np.sum((y-np.mean(y))**2)
#     r_squared = 1 - (ss_res / ss_tot)

    r_squared = calc_r_squared(x, y, out.beta, f)

    return gradient, intercept, r_squared

def xodr(ds, x, y, dim):
    
    # Check whether x and/or y have uncertainty
    unc = [f"{x}_unc", f"{y}_unc"]
    args = [ds[x], ds[y]]
    input_core_dims = [[dim]] * 2
    # loop over

    for v in unc:
        if v in ds:
            # Uncertainty found - add to arguments
            args.append(ds[v])
            input_core_dims.append([dim])
        else:
            # Uncertainty not exist - add None to arguments
            args.append(None)
            input_core_dims.append([])
    
    results = xr.apply_ufunc(
        orthoregress, *args,
        input_core_dims=input_core_dims,
        output_core_dims=[['unc'], ['unc'], []],
        vectorize=True
    )
#     results = xr.apply_ufunc(
#         orthoregress, ds[x], ds[y], ds[x + '_unc'], ds[y + '_unc'],
#         input_core_dims=[[dim], [dim], [dim], [dim]],
#         output_core_dims=[['unc'], ['unc'], []],
#         vectorize=True
#     )
    
    var_names = ['gradient', 'intercept', 'r_squared']
    unc_names = ['Measurand', 'Uncertainty']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}
    results = xr.Dataset(data_vars, coords={'unc': unc_names})

    return results

def odr_fit(f, x, y, sx, sy, beta0=None):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    
    """
    
    # ODR fit
    mod = odr.Model(f)
    dat = odr.RealData(x, y, sx=sx, sy=sy)
    od = odr.ODR(dat, mod, beta0=beta0)
    out = od.run()
    
    # Standard error of estimate
    se = calc_se(x, y, p, f)
    
    return out.beta, out.sd_beta, se

def xodr_fit(f, ds, x, y, dim, **kwargs):
    # Args f, x, y, [x_unc, y_unc]
    
    # Start args with model function
    args = [f]
    input_core_dims = [[]]
    
    # Append x and y
    args += [ds[x], ds[y]]
    input_core_dims += [[dim]] * 2
    
    # Append x and y uncertainties if exists
    unc = [f"{x}_unc", f"{y}_unc"]
    # loop over

    for v in unc:
        if v in ds:
            # Uncertainty found - add to arguments
            args.append(ds[v])
            input_core_dims.append([dim])
        else:
            # Uncertainty not exist - add None to arguments
            args.append(None)
            input_core_dims.append([])
    
    results = xr.apply_ufunc(
        odr_fit, *args,
        input_core_dims=input_core_dims,
        output_core_dims=[['p'], ['p'], []],
        kwargs=kwargs,
        vectorize=True
    )
    data_vars = {'coef': results[0], 'sd': results[1], 'se': results[2]}
    results = xr.Dataset(data_vars)

    return results

def process_fit(ds, x, y, dim, fit_method='sm'):

    # Choose fitting function
    fit_case = {
        'sm': xsmregress,
        'odr': xodr
    }
    func = fit_case.get(fit_method, None)

    if func is None:
        raise(f'{fit_method} is not in {list(fit_case.keys())}')
    
    # Fit along dim
    params = func(ds, x, y, dim)

    return params

def calc_best_fit(ds, x):
    
    best_fit = ds[x] * ds['gradient'] + ds['intercept']
    
    return best_fit

def gaussian(x,mu,sigm,amp):
    '''
    Gaussian distribution
    
    x - values for the fit
    p[0]: mu - mean of the distribution
    p[1]: sigma - stddev
    p[2]: amplitude
    '''

    return amp*np.exp(-(x-mu)**2/(2*sigm**2))

def prep_fit_results(results):
    """
    Use pybroom to prepare lmfit fit results
    """
    
    # Prep results
    param_fits = br.tidy(results).set_index('name')
    param_fits.index.name = 'param'
    param_fits = param_fits.to_xarray()

    line_fits = br.augment(results)
    line_fits = line_fits.to_xarray()

    # Merge and deliver
    results = xr.merge([param_fits, line_fits])

    return results

def gaussian_fit(ds, x, y):
    
    x = ds[x].values
    y = ds[y].values
    
    # Gaussian fit
    mod = GaussianModel(nan_policy='omit')
    pars = mod.guess(y, x=x)
    results = mod.fit(y, pars, x=x)
        
    return prep_fit_results(results)

def lorentzian_fit(ds, x, y):
    x = ds[x].values
    y = ds[y].values
    
    # Gaussian fit
    mod = LorentzianModel(nan_policy='omit')
    pars = mod.guess(y, x=x)
    results = mod.fit(y, pars, x=x)

    return prep_fit_results(results)

def rect_fit(ds, x, y):
    
    x = ds[x].values
    y = ds[y].values
    
    # Rectanglar fit
    mod = RectangleModel(nan_policy='omit')
    pars = mod.guess(y, x=x)
    results = mod.fit(y, pars, x=x)

    return prep_fit_results(results)

# def process_fit_peak_models(ds, x, y, dim):
#     gauss = process_stacked_groupby(data3, dim, gaussian_fit, x, y)
#     lorentz = process_stacked_groupby(data3, dim, lorentzian_fit, x, y)

#     fit_lines = (xr.Dataset({'raw': gauss['data'],
#                              'gauss': gauss['best_fit'],
#                              'lorentz': lorentz['best_fit']})
#                  .assign(x=gauss['x']))
#     return fit_lines

def fit_model(ds, x, y, dim, model):
    ds = ds.rename({x: 'x', y: 'y'})
    results = process_stacked_groupby(ds, dim, model, 'x', 'y')
    results['x'].attrs = ds['x'].attrs
    results['data'].attrs = ds['y'].attrs
    results['best_fit'].attrs = ds['y'].attrs
    results = results.rename({'data': 'y'})

    return results

def Linear(p,x) :
    # A linear function with:
    #   Constant Background          : p[0]
    #   Slope                        : p[1]

    return p[0]+p[1]*x

def Quadratic(p,x) :
    # A quadratic function with:
    #   Constant Background          : p[0]
    #   Slope                        : p[1]
    #   Curvature                    : p[2]

    return p[0]+p[1]*x+p[2]*x**2

def Cubic(p,x) :
    # A cubic function with:
    #   Constant Background          : p[0]
    #   Slope                        : p[1]
    #   Curvature                    : p[2]
    #   3rd Order coefficient        : p[3]

    return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3

def sm_wls(x, y, y_err):

    nobs = len(x)
    x, y, y_err = dropna(x, y, y_err)

    if len(x) > 0:

        weights = 1. / (y_err ** 2)

        # Fit data
        mod = sm.WLS(y, sm.add_constant(x), weights)

        # Results
        res = mod.fit(cov_type='fixed scale')
        coeff = res.params
        error = res.bse
        pvals = res.pvalues
        rsquared = res.rsquared
        rsquared_adj = res.rsquared_adj
        prediction = res.get_prediction().summary_frame()
        pred = prediction['mean']
        ci_lower = prediction['mean_ci_lower']
        ci_upper = prediction['mean_ci_upper']
        resid = res.resid
        # print(res.summary().tables[1])
    else:
        coeff = error = pvals = [np.nan] * 2
        rsquared = rsquared_adj = np.nan
        pred = ci_lower = ci_upper = resid = [np.nan] * nobs

    return (coeff, error, pvals,
            rsquared, rsquared_adj,
            pred, ci_lower, ci_upper, resid)

def xsm_wls(ds, x, y, y_err, dim):
    dim_ = f"{dim}_"
    ds_ = ds.swap_dims({dim: dim_})

    param = [['param']] * 3
    other = [[]] * 2
    obs_dim = [[dim]] * 4
    output_core_dims = param + other + obs_dim

    results = xr.apply_ufunc(
        sm_wls, ds_[x], ds_[y], ds_[y_err],
        input_core_dims=[[dim_], [dim_], [dim_]],
        output_core_dims=output_core_dims,
        vectorize=True
    )

    var_names = ['coeff', 'stderr', 'pvals',
                 'rsquared', 'rsquared_adj',
                 'pred', 'ci_lower', 'ci_upper', 'resid']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}

    # Merge data
    ds2 = ds.assign(data_vars)

    # Information about dimension is lost so copy informaiton
    ds2[dim] = ds[dim]

    return ds2

def sm_ols(x, y):

    nobs = len(x)
    x, y = dropna(x, y)

    if len(x) > 0:

        # Fit data
        mod = sm.OLS(y, sm.add_constant(x))

        # Results
        res = mod.fit()
        coeff = res.params
        error = res.bse
        pvals = res.pvalues
        rsquared = res.rsquared
        rsquared_adj = res.rsquared_adj
        prediction = res.get_prediction().summary_frame()
        pred = prediction['mean']
        ci_lower = prediction['mean_ci_lower']
        ci_upper = prediction['mean_ci_upper']
        resid = res.resid
        # print(res.summary().tables[1])
    else:
        coeff = error = pvals = [np.nan] * 2
        rsquared = rsquared_adj = np.nan
        pred = ci_lower = ci_upper = resid = [np.nan] * nobs

    return (coeff, error, pvals,
            rsquared, rsquared_adj,
            pred, ci_lower, ci_upper, resid)

def xsm_ols(ds, x, y, dim):
    dim_ = f"{dim}_"
    ds_ = ds.swap_dims({dim: dim_})

    param = [['param']] * 3
    other = [[]] * 2
    obs_dim = [[dim]] * 4
    output_core_dims = param + other + obs_dim

    results = xr.apply_ufunc(
        sm_ols, ds_[x], ds_[y],
        input_core_dims=[[dim_], [dim_]],
        output_core_dims=output_core_dims,
        vectorize=True
    )

    var_names = ['coeff', 'stderr', 'pvals',
                 'rsquared', 'rsquared_adj',
                 'pred', 'ci_lower', 'ci_upper', 'resid']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}

    # Merge data
    ds2 = ds.assign(data_vars)

    # Information about dimension is lost so copy informaiton
    ds2[dim] = ds[dim]

    return ds2

def sm_poly2_ols(x, y):

    nobs = len(x)
    x, y = dropna(x, y)

    if len(x) > 0:

        # Fit data
        # mod = sm.OLS(y, sm.add_constant(x))

        olsdata = {"x": x, "y": y}
        formula = "y ~ x + I(x**2)"
        mod = smf.ols(formula, data=olsdata)

        # Results
        res = mod.fit()
        coeff = res.params
        error = res.bse
        pvals = res.pvalues
        rsquared = res.rsquared
        rsquared_adj = res.rsquared_adj
        prediction = res.get_prediction().summary_frame()
        pred = prediction['mean']
        ci_lower = prediction['mean_ci_lower']
        ci_upper = prediction['mean_ci_upper']
        resid = res.resid
        # print(res.summary().tables[1])
    else:
        coeff = error = pvals = [np.nan] * 2
        rsquared = rsquared_adj = np.nan
        pred = ci_lower = ci_upper = resid = [np.nan] * nobs

    return (coeff, error, pvals,
            rsquared, rsquared_adj,
            pred, ci_lower, ci_upper, resid)

def xsm_poly2_ols(ds, x, y, dim=None):
    dim_ = f"{dim}_"
    ds_ = ds.swap_dims({dim: dim_})

    param = [['param']] * 3
    other = [[]] * 2
    obs_dim = [[dim]] * 4
    output_core_dims = param + other + obs_dim

    results = xr.apply_ufunc(
        sm_poly2_ols, ds_[x], ds_[y],
        input_core_dims=[[dim_], [dim_]],
        output_core_dims=output_core_dims,
        vectorize=True
    )

    var_names = ['coeff', 'stderr', 'pvals',
                 'rsquared', 'rsquared_adj',
                 'pred', 'ci_lower', 'ci_upper', 'resid']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}
    ds = ds.assign(data_vars)

    # Params
    ds['param'] = ['p_0', 'p_1', 'p_2']

    return ds

def sm_quadratic_ols(x, y):

    nobs = len(x)
    x, y = dropna(x, y)

    if len(x) > 0:

        # Fit data
        # mod = sm.OLS(y, sm.add_constant(x))

        olsdata = {"x": x, "y": y}
        formula = "y ~ I(x**2)"
        mod = smf.ols(formula, data=olsdata)

        # Results
        res = mod.fit()
        coeff = res.params
        error = res.bse
        pvals = res.pvalues
        rsquared = res.rsquared
        rsquared_adj = res.rsquared_adj
        prediction = res.get_prediction().summary_frame()
        pred = prediction['mean']
        ci_lower = prediction['mean_ci_lower']
        ci_upper = prediction['mean_ci_upper']
        resid = res.resid
        # print(res.summary().tables[1])
    else:
        coeff = error = pvals = [np.nan] * 2
        rsquared = rsquared_adj = np.nan
        pred = ci_lower = ci_upper = resid = [np.nan] * nobs

    return (coeff, error, pvals,
            rsquared, rsquared_adj,
            pred, ci_lower, ci_upper, resid)

def xsm_quadratic_ols(ds, x, y, dim=None):
    dim_ = f"{dim}_"
    ds_ = ds.swap_dims({dim: dim_})

    param = [['param']] * 3
    other = [[]] * 2
    obs_dim = [[dim]] * 4
    output_core_dims = param + other + obs_dim

    results = xr.apply_ufunc(
        sm_quadratic_ols, ds_[x], ds_[y],
        input_core_dims=[[dim_], [dim_]],
        output_core_dims=output_core_dims,
        vectorize=True
    )

    var_names = ['coeff', 'stderr', 'pvals',
                 'rsquared', 'rsquared_adj',
                 'pred', 'ci_lower', 'ci_upper', 'resid']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}
    ds = ds.assign(data_vars)

    # Params
    ds['param'] = ['p_0', 'p_2']

    return ds

def sm_rlm(x, y):

    nobs = len(x)
    x, y = dropna(x, y)

    if len(x) > 0:

        # Fit data
        mod = sm.RLM(y, sm.add_constant(x))

        # Results
        res = mod.fit()
        coeff = res.params
        error = res.bse
        pvals = res.pvalues
        pred = res.fittedvalues
        ci = res.conf_int()
        ci_lower = res.model.predict(ci[:, 0])
        ci_upper = res.model.predict(ci[:, 1])
        resid = res.resid
        # print(res.summary().tables[1])
    else:
        coeff = error = pvals = [np.nan] * 2
        pred = ci_lower = ci_upper = resid = [np.nan] * nobs

    return (coeff, error, pvals,
            pred, ci_lower, ci_upper, resid)

def xsm_rlm(ds, x, y, dim):
    dim_ = f"{dim}_"
    ds_ = ds.swap_dims({dim: dim_})

    param = [['param']] * 3
    obs_dim = [[dim]] * 4
    output_core_dims = param + obs_dim

    results = xr.apply_ufunc(
        sm_rlm, ds_[x], ds_[y],
        input_core_dims=[[dim_], [dim_]],
        output_core_dims=output_core_dims,
        vectorize=True
    )

    var_names = ['coeff', 'stderr', 'pvals',
                 'pred', 'ci_lower', 'ci_upper', 'resid']
    data_vars = {var_names[i]: results[i] for i in range(len(var_names))}
    ds = ds.assign(data_vars)

    return ds

def extract_params(ds, variables, attrs):
    """
    Transform coefficient and their stderr into dataarray with unc dimension
    and metadata

    Args:
        ds (xarray.Dataset): Dataset onctaining coeff and stderr dataarrays
        variables (list of str): List of quantity names
        attrs (dict): Dict of variable: dict of variable metadata.

    Return:
        xarray.Dataset: Dataset of quantities as dataarrays with unc dimension
    """
    # Reshape
    da = unc_dim(ds['coeff'], ds['stderr'])
    da['param'] = variables
    ds = da.to_dataset(dim='param')

    for k in ds.data_vars.keys():
        ds[k].attrs = attrs[k]

    return ds

def process_curve_fit(da, func_fit, xdata):
    """
    Process xarray.DataArray through scipy.curve_fit with supplied fitting
    function and selected data.
    The DataArray is dependent variable while the coords are independent
    variables.
    This function allows one or ore indepedent variables to be selected
    provided that the fiting function requires this.

    Args:
        da (xarray.DataArray): DataArray onctaining ydata and xdata
        func_fit (callable): Function to fit
        xdata (str or lisr of str): Name(s) of indepedent arrays
    """

    # Import indepedent and dependent variables
    df = (da.to_dataframe()
          .dropna()
          .reset_index())
    xdata = df[xdata].values.transpose()
    ydata = df[da.name].values
    # Fit

    p, pcov = optimize.curve_fit(func_fit, xdata, ydata)

    # Error
    perror = np.sqrt(np.diag(pcov))

    # Return parameter results
    coords = {'unc': ['Measurand', 'Uncertainty']}
    params = xr.DataArray(data=[p, perror], coords=coords, dims=['unc', 'param'])

    return params

def prediction_analysis(ds, meas, unc):
    """
    Calculate the residuals and confidence intervals from results of a model
    evaluation

    Args:
        ds (xarray.Dataset): Data
        meas (xarray.DataArray): Measurand
        unc (xarray.Dataset): Uncertainty
    """

    # Confidence intervals
    k = 2
    ci_lower = meas - unc * k
    ci_upper = meas + unc * k

    # Residuals
    resid = ds['data'] - meas

    return ds.assign(pred=meas,
                     pred_unc=unc,
                     ci_lower=ci_lower,
                     ci_upper=ci_upper,
                     resid=resid)

def process_sm_ols(ds, func_features, y=None):
    """
    Process xarray.DataArray through statsmodel.OLS with supplied feature
    transformation function.
    The DataArray is dependent variable while the coords are features and
    is transformed into full feature set with the supplied feature
    transformed function.
    The dependent variable and features will be fitted with ordinary least
    squares with statsmodel.OLS.

    Args:
        ds (xarray.DataArray or xarray.Dataset): DataArray onctaining ydata and xdata
        func_features (callable): Feature transformation function
    
    Returns:
        xarray.Dataset: Results of OLS fit added to the input DataArray.
    """

    # Stack observations
    ds2 = ds.stack(obs=[...]).dropna(dim='obs')

    # Import indepedent and dependent variables
    df = ds2.to_dataframe().reset_index()

    # Dependent variable

    if y:
        y = df[y]
    else:
        y = df[ds.name]
        ds = ds.to_dataset()

    # Independent variable(s)
    x = func_features(df)

    # Fit data
    mod = sm.OLS(y, x)

    # Results
    res = mod.fit()
    coeff = res.params
    error = res.bse
    pvals = res.pvalues
    rsquared = res.rsquared
    rsquared_adj = res.rsquared_adj
    prediction = res.get_prediction().summary_frame()
    pred = prediction['mean']
    ci_lower = prediction['mean_ci_lower']
    ci_upper = prediction['mean_ci_upper']
    resid = res.resid

    params = xr.Dataset({'coeff': ('param', coeff),
                         'stderr': ('param', error),
                         'pvals': ('param', pvals)},
                        coords={'param': list(x.columns)})

    rsquare = xr.Dataset({'rsquared': rsquared,
                          'rsquared_adj': rsquared_adj})

    # Convert to dataset

    if hasattr(ds2, 'to_dataset'):
        ds2 = ds2.to_dataset()

    pred = ds2.assign({'pred': ('obs', pred),
                       'ci_lower': ('obs', ci_lower),
                       'ci_upper': ('obs', ci_upper),
                       'resid': ('obs', resid)})

    ds2 = xr.merge([params, rsquare, pred])

    ds2 = ds2.unstack('obs')

    for k, v in ds.coords.items():
        ds2[k].attrs = v.attrs

    return ds2

def process_sm_wls(ds, func_features, y, yerr):
    """
    Process xarray.DataArray through statsmodel.OLS with supplied feature
    transformation function.
    The DataArray is dependent variable while the coords are features and
    is transformed into full feature set with the supplied feature
    transformed function.
    The dependent variable and features will be fitted with ordinary least
    squares with statsmodel.OLS.

    Args:
        ds (xarray.DataArray or xarray.Dataset): DataArray onctaining ydata and xdata
        func_features (callable): Feature transformation function
    
    Returns:
        xarray.Dataset: Results of OLS fit added to the input DataArray.
    """

    # Stack observations
    ds2 = ds.stack(obs=[...]).dropna(dim='obs')

    # Import indepedent and dependent variables
    df = ds2.to_dataframe().reset_index()

    # Dependent variable
    y = df[y]
    weights = 1. / (df[yerr] ** 2)

    # Independent variable(s)
    x = func_features(df)

    # Fit data
    mod = sm.WLS(y, x, weights)

    # Results
    res = mod.fit()
    coeff = res.params
    error = res.bse
    pvals = res.pvalues
    rsquared = res.rsquared
    rsquared_adj = res.rsquared_adj
    prediction = res.get_prediction().summary_frame()
    pred = prediction['mean']
    ci_lower = prediction['mean_ci_lower']
    ci_upper = prediction['mean_ci_upper']
    resid = res.resid

    params = xr.Dataset({'coeff': ('param', coeff),
                         'stderr': ('param', error),
                         'pvals': ('param', pvals)},
                        coords={'param': list(x.columns)})

    rsquare = xr.Dataset({'rsquared': rsquared,
                          'rsquared_adj': rsquared_adj})

    # Convert to dataset

    if hasattr(ds2, 'to_dataset'):
        ds2 = ds2.to_dataset()

    pred = ds2.assign({'pred': ('obs', pred),
                       'ci_lower': ('obs', ci_lower),
                       'ci_upper': ('obs', ci_upper),
                       'resid': ('obs', resid)})

    ds2 = xr.merge([params, rsquare, pred])

    ds2 = ds2.unstack('obs')

    for k, v in ds.coords.items():
        ds2[k].attrs = v.attrs

    return ds2

def process_sm_rlm(ds, func_features, y=None):
    """
    Process xarray.DataArray through statsmodel.RLM with supplied feature
    transformation function.
    The DataArray is dependent variable while the coords are features and
    is transformed into full feature set with the supplied feature
    transformed function.
    The dependent variable and features will be fitted with ordinary least
    squares with statsmodel.OLS.

    Args:
        ds (xarray.DataArray or xarray.Dataset): DataArray onctaining ydata and xdata
        func_features (callable): Feature transformation function

    Returns:
        xarray.Dataset: Results of OLS fit added to the input DataArray.
    """

    # Stack observations
    ds2 = ds.stack(obs=[...]).dropna(dim='obs')

    # Import indepedent and dependent variables
    df = ds2.to_dataframe().reset_index()

    # Dependent variable

    if y:
        y = df[y]
    else:
        y = df[ds.name]
        ds = ds.to_dataset()

    # Independent variable(s)
    x = func_features(df)

    # Fit data
    mod = sm.RLM(y, x)

    # Results
    res = mod.fit()
    coeff = res.params
    error = res.bse
    pvals = res.pvalues
    pred = res.fittedvalues
    resid = res.resid

    # Get more measures
    wls_res = sm.WLS(mod.endog, mod.exog, weights=mod.weights).fit()
    prstd, ci_lower, ci_upper = wls_prediction_std(wls_res)
    rsquared = wls_res.rsquared
    rsquared_adj = wls_res.rsquared_adj

    # Package into xarray
    params = xr.Dataset({'coeff': ('param', coeff),
                         'stderr': ('param', error),
                         'pvals': ('param', pvals)},
                        coords={'param': list(x.columns)})

    rsquare = xr.Dataset({'rsquared': rsquared,
                          'rsquared_adj': rsquared_adj})

    # Convert to dataset

    if hasattr(ds2, 'to_dataset'):
        ds2 = ds2.to_dataset()

    pred = ds2.assign({'pred': ('obs', pred),
                       'ci_lower': ('obs', ci_lower),
                       'ci_upper': ('obs', ci_upper),
                       'resid': ('obs', resid)})

    ds2 = xr.merge([params, rsquare, pred])

    ds2 = ds2.unstack('obs')

    for k, v in ds.coords.items():
        ds2[k].attrs = v.attrs

    return ds2

def transform_params(ds):
    """
    Transform coefficient and their stderr into dataarray with unc dimension

    Args:
        ds (xarray.Dataset): Dataset onctaining coeff and stderr dataarrays

    Return:
        xarray.Dataset: Dataset of quantities as dataarrays with unc dimension
    """
    # Reshape
    da = unc_dim(ds['coeff'], ds['stderr'])
    ds = da.to_dataset(dim='param')

    return ds
