import pandas as pd

def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    return df.to_latex(index=False, caption=caption, label=label)
