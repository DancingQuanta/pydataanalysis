---
jupyter:
  jupytext:
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

# Robust Linear Models

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
```

## Estimation

Load data:

```python
data = sm.datasets.stackloss.load()
data.exog = sm.add_constant(data.exog)
```

Huber's T norm with the (default) median absolute deviation scaling

```python
huber_t = sm.RLM(data.endog, data.exog, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
print hub_results.params
print hub_results.bse
print hub_results.summary(yname='y',
            xname=['var_%d' % i for i in range(len(hub_results.params))])
```

Huber's T norm with 'H2' covariance matrix

```python
hub_results2 = huber_t.fit(cov="H2")
print hub_results2.params
print hub_results2.bse
```

Andrew's Wave norm with Huber's Proposal 2 scaling and 'H3' covariance matrix

```python
andrew_mod = sm.RLM(data.endog, data.exog, M=sm.robust.norms.AndrewWave())
andrew_results = andrew_mod.fit(scale_est=sm.robust.scale.HuberScale(), cov="H3")
print 'Parameters: ', andrew_results.params
```

See ``help(sm.RLM.fit)`` for more options and ``module sm.robust.scale`` for scale options

## Comparing OLS and RLM

Artificial data with outliers:

```python
nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, (x1-5)**2, np.ones(nsample)]
sig = 0.3   # smaller error variance makes OLS<->RLM contrast bigger
beta = [0.5, -0.0, 5.]
y_true2 = np.dot(X, beta)
y2 = y_true2 + sig*1. * np.random.normal(size=nsample)
y2[[39,41,43,45,48]] -= 5   # add some outliers (10% of nsample)
```

### Example 1: quadratic function with linear truth

Note that the quadratic term in OLS regression will capture outlier effects. 

```python
res = sm.OLS(y2, X).fit()
print res.params
print res.bse
print res.predict
```

Estimate RLM:

```python
resrlm = sm.RLM(y2, X).fit()
print resrlm.params
print resrlm.bse
```

Draw a plot to compare OLS estimates to the robust estimates:

```python
plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(res)
plt.plot(x1, res.fittedvalues, 'r-')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')
plt.plot(x1, resrlm.fittedvalues, 'g.-')
plt.title('blue: true,   red: OLS,   green: RLM')
```

### Example 2: linear function with linear truth

Fit a new OLS model using only the linear term and the constant:

```python
X2 = X[:,[0,2]] 
res2 = sm.OLS(y2, X2).fit()
print res2.params
print res2.bse
```

Estimate RLM:

```python
resrlm2 = sm.RLM(y2, X2).fit()
print resrlm2.params
print resrlm2.bse
```

Draw a plot to compare OLS estimates to the robust estimates:

```python
prstd, iv_l, iv_u = wls_prediction_std(res2)
plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')
plt.plot(x1, res2.fittedvalues, 'r-')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')
plt.plot(x1, resrlm2.fittedvalues, 'g.-')
plt.title('blue: true,   red: OLS,   green: RLM')
```
