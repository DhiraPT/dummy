from typing import TypedDict
import pandas as pd


class Asset(TypedDict):
    name: str
    data: dict[str, pd.DataFrame]
