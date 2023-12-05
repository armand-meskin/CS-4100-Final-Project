import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

data_dict = {'time': [],
             'open': [],
             'high': [],
             'low': [],
             'close': [],
             'volume': []}
for y in range(2018, 2024):
    for i in tqdm(range(1, 13)):
        url = (f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=PEP&interval=1min'
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


def sliding_window(raw, dates, chunk_size, m_delta):
    X = []
    y = []
    d = []

    for i in tqdm(range(0, len(raw))):
        if i + chunk_size + m_delta > len(raw) - 1:
            break
        window = [raw[j + i] for j in range(0, chunk_size, m_delta)]
        X.append(window)
        y.append(raw[i + chunk_size + m_delta])
        d.append(dates[i + chunk_size + m_delta])

    return np.array(X), np.array(y), np.array(d)


raw = np.array(stocks_df['close'])

X, y, d = sliding_window(raw, np.array(stocks_df['time']), 7800, 60)

final_dict = {'time': d, 'target': y}

for i in tqdm(range(len(X))):
    for j in range(len(X[i])):
        if i == 0:
            final_dict[f'feature {j}'] = [X[i][j]]
        else:
            final_dict[f'feature {j}'].append(X[i][j])

pd.DataFrame(final_dict).to_csv('pepsi_data.csv', index_label='index')
