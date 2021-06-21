Example program that models forced oscillations
where the external force acts only for a certain time.

The script was in 2017 translated by Sebastian G. Winther-Larsen from a matlab script originally written by Arnt Inge Vistnes

```python
import numpy as np
from matplotlib import pyplot as plt
```

```python
# Importing solver
from pydataanalysis.rungekutta4 import *
```

```python
# Constants and parameters
omega = 100
Q = 25
m = 1.0e-2
k = m*omega*omega
b = m*omega/Q
F = 40
T = 6.0
impact_time = T*0.1
params = {  "A"   : b/m,
            "B"   : omega*omega,
            "C"   : F/m,
            "D"   : omega,
            "end" : impact_time,
            "func": forced_oscillation }
```

```python
# Number of steps and step size
N = 2e4
delta_t = T/N
```

```python
# Allocating arrays
y = np.zeros(int(N))
v = np.zeros(int(N))
t = np.zeros(int(N))
```

```python
for i in range(int(N-1)):

    y[i+1], v[i+1], t[i+1] = rk4r(y[i], v[i], t[i], delta_t, params)
```


```python
# Divide up data by impact event
impact_period = t < impact_time
t_i, t_d = t[impact_period], t[~impact_period]
y_i, y_d = y[impact_period], y[~impact_period]
v_i, v_d = v[impact_period], v[~impact_period]
```

```python
# Plotting
plt.plot(t_i, y_i, label='Impact')
plt.plot(t_d, y_d, label='Decay')
plt.xlabel("Time (arbitrary unit)")
plt.ylabel("Oscillation (arbitrary unit)")
plt.xlim([-0.2, T])
plt.ylim([-np.max(y)*1.2, np.max(y)*1.2])
plt.title(str(params["func"].__name__))
plt.legend()
```

```python
# Plotting
plt.plot(y, v)
plt.plot(y_d, v_d, label='Decay', color='b')
plt.plot(y_i, v_i, label='Impact', color='y')
plt.xlabel("Oscillation (arbitrary unit)")
plt.ylabel("Velocity")
plt.legend()
```

```python

```
