---
jupyter:
  jupytext:
    formats: markdown//md,notebooks//ipynb,scripts//py:percent
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

# Xarray tricks


#### Reverse a DataArray's index

```python
# Assign reversed index
da = da.reindex(z_index=da['z_index'].values[::-1])
```
