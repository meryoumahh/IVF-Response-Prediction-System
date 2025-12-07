import pandas as pd
import numpy as np

def predicting_AMH_from_n_follicules(df):
    clean_data = df.dropna(subset=['n_Follicles', 'AMH'])
    if not clean_data.empty:
        x = clean_data['n_Follicles']
        y = clean_data['AMH']
        slope, intercept = np.polyfit(x, y, 1)
        print(f"Regression Line: AMH = ({slope:.4f} * n_Follicles) + {intercept:.4f}")
    return slope, intercept

