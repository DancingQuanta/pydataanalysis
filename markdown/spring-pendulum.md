# Numerical simulation of motion of mass on spring pendulum.

Objective: Use fourth-order Runge–Kutta method to calculate damped harmonic
motion of a spring pendulum. Solve different order differential equations using
the fourth-order Runge–Kutta method.

Exercise 4 (a) from 4.4 Exercises of Physics of Oscillations and Waves

Notes:

* Pg 18 describe physics of spring pendulum.
* Pg 65 describe Runge–Kutta Method
* Written spring pendulum physics and Runge–Kutta Method into the jupyter
  notebook.
* Analyse various damping situations

## Spring pendulum

A spring pendulum is a mass hanging vertically from a fixed point with a spring.
When the spring do not have a mass attached at its bottom's edge the length of
the spring is $L_0$. When the mass is attached, the spring will stretch to a new
length $L_1$ as the force of  gravity $F_g$ pulls the mass downwards.
$$
F_s = F_g
$$
where $F_s$ is the restoring force from the spring.
Using Hookes' law to describe the spring restoring force against the gravity
$$
k (L_1 - L_0) = mg
$$
When the mass is pulled downwards slightly and released, it will oscillate about
the equilibrium point with an instantaneous displacement $L (t)$ at time $t$.
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
The differential equation is second order homogeneous differential equation with
constant coefficients.
The general solution for it is
$$
y (y) = B \sin(\omega t) + C \cos(\omega t)
$$

Now the air friction needs to be accounted for. A basic model of force from air
friction is
$$
F_F = - b v - D v^2
$$
This model is challenging to derive an general analytical solution.  At slow
speed the term $D v^2$ can be very small compared with $b v$ and so the former
term can be neglected.
$$
F_F = - b v
$$
The total sum of forces acting on the osciallting mass is thus
$$
\Sigma F = ma = m \ddot{y} \\
- k y (t) - b \dot{y} (t) = m \ddot{y} (t) \\
\frac{k}{m} y (t) + \frac{b}{m} \dot{y} (t) + \ddot{y} (t) = 0 \\
$$

This is a homogeneous second-order differential equation. Choosing a trial
solution of the type:
$$
y (t) = A e^{\alpha t}
$$
Insertion of this trial solution into the differential equation gives a
polynomial
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


This solution gives three different physical situations:

* $\gamma > \omega$ Overdamping
* $\gamma < \omega$ Underdamping
* $\gamma = \omega$ Critical damping

## Overdamping

Define overdamping coefficient
$$
\alpha = \sqrt{\gamma^2 - \omega^2}
$$
Displacement over time
$$
z(t) = A_1 \exp{(-\gamma + \alpha) t} +
A_2 \exp{(-\gamma - \alpha) t}
$$
Initial conditions:
$$
z(0) = x_0 = A_1 + A_2
$$
$$
v(t=0) = v_0 = (-\gamma + \alpha) A_1 + (-\gamma - \alpha) A_2 = a A_1 + b A_2
$$
$$
A_2 = \frac{v_0 - a x_0}{b-a}
$$
$$
A_1 = x_0 - A_2
$$

## Underdamping

$$
\omega' = \sqrt{\omega^2 - \gamma^2}
$$
$$
z(t) = A \exp{(-\gamma t)} \cos{(\omega' t + \phi)}
$$

$$
A = \left( x_0^2 + \left(\frac{v_0+\gamma c_0}{\omega'} \right)^2 \right)^{1/2}
$$


Initial conditions:

* m = 100g = 0.1 kg
* k = 10N/m
* b = 0.10kg/s
* z(0) = 10 cm = 0.1 m
* v (t=0) = 0 m/s

```python
# Constants and parameters

m = 0.1
k = 10
b = 0.1

# Initial conditions
y_0 = 0.1
v_0 = 0

import numpy as np

gamma = b / (m * 2)
omega = np.sqrt(k/m)
```

```python
def oscillation_case(gamma, omega):
    if gamma > omega:
        return 'overdamped'
    elif gamma < omega:
        return 'underdamped'
    elif gamma == omega:
        return 'critical'

{'case': oscillation_case(gamma, omega), 'gamma': gamma, 'omega': omega}
```

```python
# Analytical solution using the trial solution
# page 18

def overdamped_initial(gamma, omega, x_0, v_0):
    determinant = np.sqrt(gamma**2 - omega**2)

    a = - gamma + determinant
    b = - gamma - determinant
    A_2 = (v_0 - a x_0) / (b - a)
    A_1 = x_0 - A_2

    return A_1, A_2

def overdamped_solution(t, gamma, omega, A_1, A_2):
    determinant = np.sqrt(gamma**2 - omega**2)
    return (A_1 * np.exp((-gamma + determinant)*t) +
            A_2 * np.exp((-gamma - determinant)*t))

def criticaldamped_solution(t, gamma, A, B):
    return (A * np.exp(-gamma * t) +
            B * t * np.exp(-gamma * t))

def underdamped_solution(t, gamma, omega, phi):
    reduced_omega = np.sqrt(omega**2 - gamma**2)
    return (A * np.exp(-gamma * t) *
            np.cos(reduced_omega*t + phi))
```

```python
# Plotting
y = overdamped_solution(t, gamma, omega, A_1, A_2)

plt.plot(t, y)
plt.xlabel("Time (arbitrary unit)")
plt.ylabel("Oscillation (arbitrary unit)")
plt.xlim([-0.2, T])
plt.ylim([-np.max(y)*1.2, np.max(y)*1.2])
plt.title(str(params["func"].__name__))
```

## Runge–Kutta method

The equation of motion of mass-spring pendulum is
$$
a (t) = - \frac{b}{m} v (t) - \frac{k}{m} y (t)
$$
where $a (t) = \ddot{y} (t)$ is acceleration and
$v = \dot{y} (t)$ is velocity.

This needs to be converted into two coupled first-order differential equations.
First a change of variables
$$
\begin{aligned}
x (t) &= y (t) \\
v (t) &= \frac{d y}{d t} \\
\end{aligned}
$$
and then differentiate both sides
$$
\begin{aligned}
\frac{d x}{d t} &= \frac{d y}{d t} = v (y (t), t) \\
\frac{d v}{d t} &= \frac{d^2 y}{d t^2} = a (t) \\
\end{aligned}
$$
thus the coupled first order differential equations are
$$
\begin{aligned}
\frac{d x}{d t} &= v (y (t), t) \\
\frac{d v}{d t} &= - \frac{b}{m} v (t) - \frac{k}{m} y (t) \\
\end{aligned}
$$
Simplify $a (t)$
$$
\frac{d v}{d t} = - A v (t) - B y (t)
$$
where $A = \frac{b}{m}$ and $B = \frac{k}{m}$


We are at the point $(y_n, v_n, t_n)$ and the time duration step is $\Delta t$.
The Runge–Kutta method involves calculating $k=4$ estimates for $(y_n, v_n,
a_n)$.  The $k$th estimate of an quantity will be represented by $y_{k, n}$.

1. The first estimate of $a_n$ can be found by
    $$
    a_{1,n} = - \frac{b}{m} v_n (t) - \frac{k}{m} y_n (t)
    $$
    At the beginning of the time step the velocity is
    $$
    v_{1, n} = v_n
    $$
2. Use Euler's method to find $y_n$ and $v_n$ in the
    middle of the step
    $$
    \begin{aligned}
    x_{2,n} &= y_{1,n} + v_{1,n} \frac{\Delta t}{2} \\
    v_{2,n} &= y_{1,n} + a_{1,n} \frac{\Delta t}{2} \\
    \end{aligned}
    $$
    Then find an estimate of $a_{2,n}$
    $$
    a_{2,n} = f(y_{2,n}, v_{2,n}, t_n + \Delta t / 2)
    $$
3. Use Euler's method and $a_{2,n}$ to find another estimate
    at the midpoint
    $$
    \begin{aligned}
    x_{3,n} &= y_{1,n} + v_{2,n} \frac{\Delta t}{2} \\
    v_{3,n} &= v_{1,n} + a_{2,n} \frac{\Delta t}{2} \\
    \end{aligned}
    $$
    Then find an estimate of $a_{3,n}$
    $$
    a_{3,n} = f(y_{3,n}, v_{3,n}, t_n + \Delta t / 2)
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
# Importing solver
from pydataanalysis.rungekutta4 import *
```

```python
params = {  "A"   : b/m,
            "B"   : k/m,
            "func": damped_oscillation }
```

```python
# Number of steps and step size
T = 6
N = 1e4
delta_t = T/N
delta_t
```

```python
# Allocating arrays
y = np.zeros(int(N))
v = np.zeros(int(N))
t = np.zeros(int(N))
y[0] = y_0
v[0] = v_0
```

```python
for i in range(int(N-1)):

    y[i+1], v[i+1], t[i+1] = rk4r(y[i], v[i], t[i], delta_t, params)
```


```python
# Plotting
plt.plot(t, y)
plt.xlabel("Time (arbitrary unit)")
plt.ylabel("Oscillation (arbitrary unit)")
plt.xlim([-0.2, T])
plt.ylim([-np.max(y)*1.2, np.max(y)*1.2])
plt.title(str(params["func"].__name__))
```

```python
# Plotting
plt.plot(y, v)
plt.xlabel("Oscillation (arbitrary unit)")
plt.ylabel("Velocity")
```
