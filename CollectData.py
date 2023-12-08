import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

STOCK_TICKER = "SHOP"
STOCK_NAME = "shopify"

data_dict = {'time': [],
             'open': [],
             'high': [],
             'low': [],
             'close': [],
             'volume': []}
for y in range(2018, 2024):
    for i in tqdm(range(1, 13)):
        url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={STOCK_TICKER}&interval=1min'
               f'&month={y}-{str(i).zfill(2)}'
               f'&outputsize=full&adjusted=false&datatype=json&extended_hours=false&apikey=3BAAT9BW5TUZBZY8')
        r = requests.get(url)
        data = r.json()

        for t, d in data['Time Series (1min)'].items():
            data_dict['time'].append(t)
            data_dict['open'].append(d['1. open'])
            data_dict['high'].append(d['2. high'])
            data_dict['low'].append(d['3. low'])
            data_dict['close'].append(d['4. close'])
            data_dict['volume'].append(d['5. volume'])

stocks_df = pd.DataFrame(data_dict)
stocks_df.sort_values(by=['time'], inplace=True)
stocks_df.to_csv(f'{STOCK_NAME}_data.csv', index_label='index')
