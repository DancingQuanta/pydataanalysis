Numerical simulation of motion of mass on spring pendulum.

The mass is hung vertically from a fixed point with a spring.
When the spring do not have a mass attached at its botoom's edge
the length of the spring is $L_0$. When the mass is attached, the
spring will stretch to a new length $L_1$ as the force of  gravity
$F_g$ pulls the mass downwards.
$$
F_s = F_g
$$
where $F_s$ is the restoring force from the spring.
Using Hookes' law to describe the spring restoring force against the
gravity
$$
k (L_1 - L_0) = mg
$$
When the mass is pulled downwards slightly and released, it will
oscillate about the equilibrium point with an instantenous
displacement $L (t)$ at time $t$.
$$
F (t) = k [L (t) - L_0] - mg
$$
Combining the last two equations
$$
F (t) = k [L (t) - L_1]
$$
Let simplify the last expression by letting $y (t) = L_1 - L (t)$
$$
F (t) = - k y (t)
$$
Using Newton's second law
$$
ma = m \ddot{y} = - k y (t)
$$
The equation of motion is then can be written
$$
\ddot{y} = - \frac{k}{m} y (t) = - \omega^2 y (t)
$$
where
$$
\omega = \sqrt{\frac{k}{m}}
$$
is the angular frequency.
The differential equation is second order homogeneous differential
equation with constant coef-ficients. 
The general solution for it is
$$
y (y) = B \sin(\omega t) + C \cos(\omega t)
$$

now the air friction needs to be accounted for. A basic model of
force from air friction is
$$
F_F = - b v - D v^2
$$
This model is challenging to derive an general analytical solution.
At slow speed the term $D v^2$ can be very small compared with $b v$
and so the former term can be neglected.
$$
F_F = - b v
$$
The total sum of forces acting on the osciallting mass is thus
$$
\Sigma F = ma = m \ddot{y} \\
- k y (t) - b \dot{y} (t) = m \ddot{y} (t) \\
\frac{k}{m} y (t) + \frac{b}{m} \dot{y} (t) + \ddot{y} (t) = 0 \\
$$

This is a homogeneous second-order differential equation. Choosing a
trial solution of the type:
$$
y (t) = A e^{\alpha t}
$$
Insertion of this trial solution into the differential equation gives
a polynominal
$$
\alpha^2 + \frac{b}{m} \alpha + \frac{k}{m} = 0
$$
Define some relations
$$
\frac{b}{m} \equiv 2 \gamma \\
\frac{k}{m} \equiv \omega^2 \\ 
$$
and inserting them into the polynominal equation
$$
\alpha^2 + 2 \gamma \alpha + \omega^2 = 0
$$
This is a quadratic equation whose roots can be written as:
$$
\alpha_{\pm} = - \gamma \pm \sqrt{\gamma^2 - \omega^2}
$$

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
