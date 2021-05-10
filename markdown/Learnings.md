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

```python
# TODO
def sin(x, freq):
    return np.sin(2*np.pi*freq*x))

# Initialise kdims
length=0.1
sampleRate=2000
sampleInc = 1.0/sampleRate
freq=

t = xr.DataArray(np.arange(0,length, sampleInc), dims=['time'], attrs={'long_name': 'Time'})
freq = xr.DataArray(np.linspace(20, 60, 3), dims=['freq'], attrs={'long_name': 'Carrier frequency'})
f_mod = xr.DataArray(np.linspace(20, 100, 5), dims=['f_mod'], attrs={'long_name': 'Modulation frequency'})

# A list of maximum fields as a list of field regions
MaxField = [5000, 100, 20]

# Broadcast data to include maximum field
MaxField = xr.DataArray(np.ones_like(MaxField),
                        coords={'MaxField': MaxField},
                        dims=['MaxField'],
                        name='Field',
                        attrs={'long_name': 'Field'}
                       )
data = data * MaxField

# Filter the fields with maximum field by converting field
# values above maximum to NaN
data = data.where(np.abs(data.Field) <= data.MaxField, drop=True)

# Generate plot of selected variable against field split by MaxField
plot = (data.hvplot.scatter('Field', variable,
                           groupby=['MaxField'], dynamic=False)
        .layout('MaxField')
        .opts(
            opts.Scatter(title_format='M (H) curve for H < {label} mT',
                         axiswise=True
                        )
        )
       )

layout_list = []
for k, v in plot.data.items():
    idx = v.data[variable].dropna().index
    v.data = v.data.loc[idx]
    k = k[0]
    v = v.opts(title=f'M (H) curve for H < {k} mT',
               framewise=True, axiswise=True)
    layout_list.append(v)

plot = (hv.Layout(layout_list)
        .opts(opts.Scatter(axiswise=True))
       )
```
