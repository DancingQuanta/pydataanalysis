# Douglas Myers-Turnbull wrote this for the Kokel Lab, which has released it under the Apache Software License, Version 2.0
# See the license file here: https://gist.github.com/dmyersturnbull/bfa1c3371e7449db553aaa1e7cd3cac1
# The list of copyright owners is unknown

# Andrew Tolmie: some corrections

import pandas as pd
import itertools
from typing import Iterator, List

def marker_iterator(df: pd.DataFrame, class_column: str='class') -> Iterator[str]:
    """Returns an iterator of decent marker shapes. The order is such that similar markers aren't used unless they're needed."""
    # Don't use MarkerStyle.markers because some are redundant
    lst = ['o', 's', '*', 'v', '^', 'D', 'h', 'x', '+', '8', 'p', '<', '>', 'd', 'H']
    if df.groupby(class_column).count().max() > len(lst):
        warnings.warn("Currently limited to {} markers; some will be recycled".format(len(lst)))
    return itertools.cycle(lst)

def markers_for_rows(df: pd.DataFrame, class_column: str='class') -> List[str]:
    """Returns a list of markers, one for each row.
    Mostly useful as a reminder not to give Seaborn lmplot markers=a_pandas_series: it needs a list.
    """
    markers = marker_iterator(df, class_column)

    # NOTE: With Seaborn lmplot, if you just pass the Series markers_df['marker] without doing .values.tolist(), you'll get an error
    return df.index.map(lambda _: next(markers)).values.tolist()
