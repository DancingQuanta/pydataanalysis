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
import numpy as np
import pandas as pd
```

```python
df = pd.DataFrame({'A': list('aaabbbccccdd'), 'B': list('353452462341')})
df['B'] = df['B'].astype('int')
df
```

```python
dfg = df.groupby('A')
dfg.size() == 3
```

```python
df[df['A'].map(df['A'].value_counts()) == 3]
```

```python
df[df.groupby("A")['A'].transform('size') == 3]
```

```python
df.groupby("A")['B'].mean()
```
