import pandas as pd
import uncertainties

# Pandas
from pandas.io.formats import format as fmt

# Format floats

import string
class ShorthandFormatter(string.Formatter):

    def format_field(self, value, format_spec):
        if isinstance(value, uncertainties.UFloat):
            return value.format(format_spec+'S')  # Shorthand option added
        # Special formatting for other types can be added here (floats, etc.)
        else:
            # Usual formatting:
            return super(ShorthandFormatter, self).format_field(
                value, format_spec)

frmtr = ShorthandFormatter()

pd.set_option('display.float_format', lambda x: frmtr.format('{:.5g}', x))
# pd.set_option('display.float_format', '{:.4g}'.format)

# Enable latex
pd.options.display.latex.repr = True

def _repr_latex_(self):
    return (r"\begin{center}"
            "\n%s\n"
            r"\end{center}") % self.to_latex(index=True, escape=False)

def _repr_markdown_(self):
    import tabulatehelper as th
    return f"{th.md_table(self)}"

print(pd.DataFrame._repr_latex_)
pd.DataFrame._repr_latex_ = _repr_latex_  
# pd.DataFrame._repr_markdown_ = _repr_markdown_
# pd.DataFrame._repr_html_ = _repr_html_