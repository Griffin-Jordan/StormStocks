import pandas as pd
import numpy as np

aapl = pd.read_csv("/Users/JerryStrippoli/Downloads/DS440/FrontEnd/data/AAPL.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
aapl["ticker"] = "AAPL"

msft = pd.read_csv("/Users/JerryStrippoli/Downloads/DS440/FrontEnd/data/MSFT.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
msft["ticker"] = "MSFT"


df = pd.concat([aapl, msft], ignore_index=True)
df

df.to_csv('/Users/JerryStrippoli/Downloads/DS440/FrontEnd/data/stocks.csv', index=False)

