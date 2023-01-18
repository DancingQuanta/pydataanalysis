Box-Muller transform
====================

Explore the Box-Muller algorithm, by generating and plotting the results.

The algorithm generates a pair of standard normal random variables from a
uniformly random variable. Two samples were taken from the uniform distribution
of an interval $[0, 1]$ and maps them to two standard, normally distributed
samples. This is the basic form involving trigonometric functions. A second
polar form takes two samples from a uniform distribution of an interval $[-1,
1]$ and maps them to two normally distributed samples without trigonometric
functions.

Basic form
----------

Two samples, $U_1$ and $U_2$, are drawn from the uniform distribution on the
unit interval $[0, 1]$. These samples will be mapped to $Z_0 and $Z_1$ through
the following equations;
$$
Z_0 = R \cos(\Theta) = \sqrt{-2 \ln U_1} \cos(2 \pi U_2)\,
$$
and
$$
Z_1 = R \sin(\Theta) = \sqrt{-2 \ln U_1} \sin(2 \pi U_2).\,
$$

Polar form
----------

TODO:

* Study properties of basic form
* Simulate basic form

References:

* [Box-Muller transform - Wikipedia](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
