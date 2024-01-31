# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:33:44 2024

@author: bongw
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.pandas.read_csv("time_series_data.csv")

print(df.info())

df['Date']=pd.to_datetime(df['Date'], format="%Y-%m-%d")
print(df.info())

# plt.plot(df['



