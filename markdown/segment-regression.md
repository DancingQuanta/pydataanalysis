---
jupyter:
  jupytext:
    formats: markdown//md,notebooks//ipynb,scripts//py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# def f(x, xb, y0, ks):
#     if len(xb) != len(ks)+1:
#         raise

#     yb = np.like(xb)
#     yb[0] = y0
#     for i in range(len(yb)-1)
#         yb[i+1] = yb[i] + (ks[i+1] - xb[i])

#     for i in range(len(xb)):
#         if i == 0:
#             cond = x < xb[i]
#             y = yb[i] + ks[i]
#         elif i == len(xb):
#             cond = x >= xb[i]
#         else:
#             cond = (x >= xb[i-1] & x < xb[i])



```

```python
# from scipy import optimize
# import matplotlib.pyplot as plt
# import numpy as np

# sel_data = data2.sel(I=-2, rep=4, direction='negative')
# x = sel_data['B'].values
# y = sel_data['sig1'].values

# verts = [89, 93, 158, 167]
# xn = x[verts]
# y0 = y[verts[0]]
# kn = [1] * 5
# p0 = xn + [y0] + kn

# def f(x,x0,x1,x2,x3,x4,y0,
#       k0,k1,k2,k3,k4,k5):
#     # x0,y0 : first breakpoint
#     # x1 : second breakpoint
#     # k1,k2,k3 : 3 slopes.

#     y1 = y0 + k1*(x1-x0)
#     y2 = y1 + k2*(x2-x1)
#     y3 = y2 + k3*(x3-x2)
#     y4 = y3 + k4*(x4-x3)
#     return (
#     (x<x0)              *   (y0 + k0*(x-x0))      +
#     ((x>=x0) & (x<x1))  *   (y0 + k1*(x-x0))      +
#     ((x>=x1) & (x<x2))  *   (y1 + k2*(x-x1))      +
#     ((x>=x2) & (x<x3))  *   (y2 + k3*(x-x2))      +
#     ((x>=x3) & (x<x4))  *   (y3 + k4*(x-x3))      +
#     (x>=x4)             *   (y4 + k5*(x-x4)))

# p, e = optimize.curve_fit(f, x, y,p0)
# print(p)
# print(e)
# plt.plot(x, y, "o")
# plt.plot(x,f(x,*p))
```

```python
# x = np.array([-1.280199006, -1.136209343, -1.048070216, -0.9616764178, -0.8752826199, -0.7871434926, -0.6981317008, -0.6108652382, -0.5235987756, -0.4372049776,
#  -0.3490658504, -0.2644173817, -0.1762782545,
# -0.0907571211, 0, 0.09250245036, 0.1762782545, 0.2661627109, 0.3516838443, 0.4345869837, 0.529707428, 0.6108652382, 0.7007496947, 0.7880161573, 0.872664626,
# 0.9616764178, 1.055051533, 1.160643953, 1.274090354, 1.413716694])

# y = np.array([-0.05860218717, -0.05275174988, -0.04961805822, -0.02860635697, -0.04150466841, -0.02672933264, -0.02422597285, -0.03056176732, -0.02885180089, -0.02085851636,
# -0.02873319291, -0.02374542821, -0.02132671806,
# -0.02088924602, -0.0216617248, -0.01835553738, -0.01369531698, -0.01331112368, -0.01156455074, -0.009163690404, -0.003542622659, -0.003515924976, -0.003828831726, -0.002622163805, -0.001622083468,
# -0.00297346133, -0.001845415856, -0.001913228234, -0.001495496086, -0.001454621173])

# def f(x,x0,y0,x1,k1,k2,k3):
#     # x0,y0 : first breakpoint
#     # x1 : second breakpoint
#     # k1,k2,k3 : 3 slopes.

# #     print((x<x0)   *   (y0 + k1*(x-x0)))

#     y1=y0+ k2*(x1-x0) # for continuity
#     return (
#     (x<x0)              *   (y0 + k1*(x-x0))      +
#     ((x>=x0) & (x<x1))  *   (y0 + k2*(x-x0))      +
#     (x>=x1)             *   (y1 + k3*(x-x1)))

# p0=(-.7,-0.03,.5,0.03,0.02,0.01)
# p , e = optimize.curve_fit(f, x, y,p0)
# print(p)
# plt.plot(x, y, "o")
# plt.plot(x,f(x,*p))
```

```python
# from scipy import optimize
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
# y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

# def piecewise_linear(x, x0, y0, k1, k2):
#     return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

# p , e = optimize.curve_fit(piecewise_linear, x, y)
# xd = np.linspace(0, 15, 100)
# plt.plot(x, y, "o")
# plt.plot(xd, piecewise_linear(xd, *p))
# print(p)
```

```python
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
# from scipy.signal import find_peaks

# sel_data = sel_data.sel(I=-2, rep=4, direction='negative')
# x = sel_data['B'].values
# y = sel_data['sig1'].values

# tck = interpolate.splrep(x, y, k=3, s=0.000000002)
# xnew = np.arange(x[0], x[-1], 0.1)

# fig, axes = plt.subplots(3)

# axes[0].plot(x, y, 'x', label = 'data')
# axes[0].plot(xnew, interpolate.splev(xnew, tck, der=0), label = 'Fit')
# axes[1].plot(x, interpolate.splev(x, tck, der=1), label = '1st dev')
# dev_2 = interpolate.splev(x, tck, der=2)
# axes[2].plot(x, dev_2, label = '2st dev')

# turning_point_mask = dev_2 == np.amax(dev_2)
# peaks, _ = find_peaks(dev_2)
# print(peaks)
# axes[2].plot(x[peaks], dev_2[peaks],'rx',
#              label = 'Turning point')
# peaks, _ = find_peaks(dev_2*-1)
# print(peaks)
# axes[2].plot(x[peaks], dev_2[peaks],'bx',
#              label = 'Turning point')
# for ax in axes:
#     ax.legend(loc = 'best')
```
