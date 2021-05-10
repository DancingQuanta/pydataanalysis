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

def percent_average(mx, mi, avg):
    num = mx.sel(unc='Measurand') - mi.sel(unc='Measurand')
    percent = (num / avg.sel(unc='Measurand')) * 100
    u_num = sqrt_ss([mx.sel(unc='Uncertainty'), mi.sel(unc='Uncertainty')])
    u_percent = mul_prog(percent,
                         [num, avg.sel(unc='Measurand')],
                         [u_num, avg.sel(unc='Uncertainty')])
    return percent, percent_u
    
    
nominal = mass.pipe(extract_measurand)
percent = ((nominal['Pre MPMS'] - nominal['Post MPMS']) / nominal['Average']) * 100
unc = mass.pipe(extract_unc)
num = sqrt_ss(unc[['Pre MPMS', 'Post MPMS']].to_array(sim='case').values)
unc = mul_prog(percent, nominal.to_array(dim='case').values, unc.to_array(dim='case').values)
percent = ((mass['Pre MPMS'] - mass['Post MPMS']) / mass['Average']) * 100
```

```python
sample = mass.sel(holder='Sample L2')
num = (sample['Pre MPMS'] - sample['Post MPMS'])
dem = sample['Average']
dem.item().error_components()
```

```python
nominal = sample.pipe(extract_measurand)
percent = ((nominal['Pre MPMS'] - nominal['Post MPMS']) / nominal['Average']) * 100
from src.statistics import extract_unc
unc = sample.pipe(extract_unc)
unc = mul_prog(percent, nominal.to_array(dim='case').values, unc.to_array(dim='case').values)
unc
```

```python
# df = (xr.open_dataset(raw_data_file, engine='netcdf4', group='LIQUID')
#       ['open_time']
#       .reset_coords(drop=True)
#       .to_dataframe()
#       .reset_index()
#       .set_index('open_time'))

# print(df)
# # Resample by day and count number of occurance per day
# # This have an positive effect of creating dates with no occurance
# count = df.resample('H').count()

# def pad_day(df):
#     # Pad to begin and end of months
#     idx_days = pd.date_range(start=df.index[1]-pd.offsets.YearBegin(),
#                              end=df.index[1]+pd.offsets.YearEnd(),
#                              freq='H')
#     return df.reindex(idx_days, fill_value=0)
# def pad_year(df):
#     # Pad to begin and end of months
#     idx_days = pd.date_range(start=df.index[1]-pd.offsets.YearBegin(),
#                              end=df.index[1]+pd.offsets.YearEnd(),
#                              freq='H')
#     return df.reindex(idx_days, fill_value=0)

# def pad_week(df):
#     # Pad to begin and end of months
#     start = df.index[0]
#     end = df.index[-1]
#     idx_days = pd.date_range(start=start-np.timedelta64(start.weekday(), 'D'),
#                              end=end+np.timedelta64(6-end.weekday(), 'D'),
#                              freq='H')
#     return df.reindex(idx_days, fill_value=np.nan)

# # months = np.ceil((count.index[-1]-count.index[0]) / np.timedelta64(1, 'M')).astype('int') + 1
# # weeks = np.ceil((count.index[-1]-count.index[0]) / np.timedelta64(1, 'W')).astype('int') + 1
# # # years = count.index.year.unique()

# # for year in years:
# #     print(count[count.index.year==year])
    
# # pad nan to whole week at edges
# # Split by year

# # ----------------------------------------------------------------------------
# # Author:  Nicolas P. Rougier
# # License: BSD
# # ----------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from datetime import datetime
# from dateutil.relativedelta import relativedelta


# def calmap(ax, year, data):
#     print(data)
#     ax.tick_params('x', length=0, labelsize="medium", which='major')
#     ax.tick_params('y', length=0, labelsize="x-small", which='major')

#     # Month borders
#     xticks, labels = [], []
#     start = datetime(year,1,1).weekday()
#     for month in range(1,13):
#         first = datetime(year, month, 1)
#         last = first + relativedelta(months=1, days=-1)

#         y0 = first.weekday()
#         y1 = last.weekday()
#         x0 = (int(first.strftime("%j"))+start-1)//7
#         x1 = (int(last.strftime("%j"))+start-1)//7

#         P = [ (x0,   y0), (x0,    7),  (x1,   7),
#               (x1,   y1+1), (x1+1,  y1+1), (x1+1, 0),
#               (x0+1,  0), (x0+1,  y0) ]
#         xticks.append(x0 +(x1-x0+1)/2)
#         labels.append(first.strftime("%b"))
#         poly = Polygon(P, edgecolor="black", facecolor="None",
#                        linewidth=1, zorder=20, clip_on=False)
#         ax.add_artist(poly)
    
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(labels)
#     ax.set_yticks(0.5 + np.arange(7))
#     ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
#     ax.set_title("{}".format(year), weight="semibold")
    
#     ax.set_xlim([0,53])
#     ax.set_ylim([0,7])
#     ax.set_aspect(1)
    
#     # Showing data
#     ax.imshow(data, extent=[0,53,0,7], zorder=10, vmin=-1, vmax=1,
#               cmap="RdYlBu", origin="lower", alpha=.75)


# # Split data by years
# groups = count.groupby(count.index.year)

# fig, axes = plt.subplots(len(groups), 1, figsize=(8,4.5), dpi=100)

# zipped = zip(axes, groups)
# for ax, (k, v) in zipped:
# #     v = pad_year(v)
# #     v = pad_week(v)
#     print(v)
#     print(len(v))
#     print(53*7)
#     calmap(ax, k, v.values.reshape(53,7*24).T)

# plt.tight_layout()
# plt.show()
```
