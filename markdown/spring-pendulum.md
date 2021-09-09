# Numerical simulation of motion of mass on spring pendulum.

Objective: Use fourth-order Runge–Kutta method to calculate damped
harmonic motion of a spring pendulum. Solve different order
differential equations using the fourth-order Runge–Kutta method.

Exercise 4 (a) from 4.4 Exercises of Physics of Oscillations and Waves

Notes:

* Pg 18 describe physics of spring pendulum.
* Pg 65 describe Runge–Kutta Method
* Written spring pendulum physics and Runge–Kutta Method into the jupyter
  notebook.

## Spring pendulum

A spring pendulum is a mass hanging vertically from a fixed point
with a spring.
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


## Runge–Kutta method

The equation of motion of mass-spring pendulum is
$$
a (t) = - \frac{b}{m} v (t) - \frac{k}{m} y (t)
$$
where $a (t) = \ddot{y} (t)$ is acceleration and
$v = \dot{y} (t)$ is velocity.

Converting into two coupled first-order differential
equations
$$
\frac{d x}{d t} = v (x (t), t)
$$
$$
\frac{d v}{d t} = - \frac{b}{m} v (t) - \frac{k}{m} y (t)
$$

We are at the point $(x_n, v_n, t_n)$ and the time
duration step is $\Delta t$. The Runge–Kutta method
involves calculating $k=4$ estimates for $(x_n, v_n, a_n)$.
The $k$th estimate of an quantity will be represented by
$x_{k, n}$.

1. The first estimate of $a_n$ can be found by
    $$
    a_{1,n} = f(x_n, v_n, t_n)
    $$
    At the beginning of the time step the velocity is
    $$
    v_{1, n} = v_n
    $$
2. Use Euler's method to find $x_n$ and $v_n$ in the
    middle of the step
    $$
    \begin{aligned}
    x_{2,n} &= x_{1,n} + v_{1,n} \frac{\Delta t}{2} \\
    v_{2,n} &= v_{1,n} + a_{1,n} \frac{\Delta t}{2} \\
    \end{aligned}
    $$
    Then find an estimate of $a_{2,n}$
    $$
    a_{2,n} = f(x_{2,n}, v_{2,n}, t_n + \Delta t / 2)
    $$
3. Use Euler's method and $a_{2,n}$ to find another estimate
    at the midpoint
    $$
    \begin{aligned}
    x_{3,n} &= x_{1,n} + v_{2,n} \frac{\Delta t}{2} \\
    v_{3,n} &= v_{1,n} + a_{2,n} \frac{\Delta t}{2} \\
    \end{aligned}
    $$
    Then find an estimate of $a_{3,n}$
    $$
    a_{3,n} = f(x_{3,n}, v_{3,n}, t_n + \Delta t / 2)
    $$
4. Use Euler's method to find $x_n$ and $v_n$ in the
    end of the step
    $$
    \begin{aligned}
    x_{4,n} &= x_{1,n} + v_{3,n} \Delta t \\
    v_{4,n} &= v_{1,n} + a_{3,n} \Delta t \\
    \end{aligned}
    $$
    Then find an estimate of $a_{4,n}$
    $$
    a_{4,n} = f(x_{4,n}, v_{4,n}, t_n + \Delta t)
    $$
5. Calculate weigthed average of $a_n$ and $v_n$
    $$
    \bar{a_n} = \frac{1}{6} (a_{1,n} + 2 a_{2,n} + 2 a_{3,n} + a_{4,n})
    $$
    $$
    \bar{v_n} = \frac{1}{6} (v_{1,n} + 2 v_{2,n} + 2 v_{3,n} + v_{4,n})
    $$
6. Use Eular's method to find $x (t)$ and $\dot{x} (t)$
    $$
    x_{n+1} = x_n + \bar{v_n} \Delta t
    $$
    $$
    v_{n+1} = v_n + \bar{a_n} \Delta t
    $$
    $$
    t_{n+1} = t_n + \Delta t
    $$
    which can be used as the initial values for the next step.

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
