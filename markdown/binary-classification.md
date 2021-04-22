# Binary classification

Compare a set of data with a threshold value and assign True if less than and
False if greater than the threshold.

```python
binary = data > thres
```

This creates a mask which can be used to select data.

To select data with numpy

```python
lower = data[binary]
upper = data[~binary]
```
