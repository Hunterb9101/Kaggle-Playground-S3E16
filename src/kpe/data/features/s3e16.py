from typing import List, Tuple
import pandas as pd
import numpy as np

class CustomFeatures:
    def __init__(self, ratio: List[Tuple[str, str]]):
        """
        
        Parameters
        ----------
        ratio : List[Tuple[str, str]]
            A list of (fraction,total) column pairs to calculate a ratio from
        
        """
        self.ratio = ratio

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for frac, tot in self.ratio:
            df[f"{frac}_to_{tot}_ratio"] = df[frac] / df[tot]

        df["Shell_SA"] = (df["Diameter"] * df["Length"] * 2) \
            + (df["Diameter"] * df["Height"] * 2) + (df["Length"] * df["Height"] * 2)
        df["Volume"] = df["Diameter"] * df["Height"] * df["Length"]
        df["Density"] = np.clip(df["Weight"] / df["Volume"], 0, 75)
        df["BMI"] = np.clip(df["Weight"] / (df["Height"] ** 2), 0 ,75)
        # Calculate the Length-Minus-Height
        df["Length_Minus_Height"] = df["Length"] - df["Height"]

        return df
