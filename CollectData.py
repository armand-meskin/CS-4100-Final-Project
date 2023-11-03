import requests
import json
import time
import os
# Collect the past 5 years and store in csv...
# End at 2023-11 (yr-month)
# Start at 2018-11 (yr-month)
year = 2018
month = 11

while year <= 2024:
    while month != 13:
        formatMonth = month
        if formatMonth < 10:
            formatMonth = "0" + str(formatMonth)
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&month={year}-{formatMonth}&outputsize=full&adjusted=false&datatype=json&extended_hours=false&apikey=3BAAT9BW5TUZBZY8'
        r = requests.get(url)
        with open('data.json', 'w') as file:
            if os.path.getsize("data.json") == 0:
                json.dump(r.json()["Time Series (1min)"], file)
            else:
                file_data = json.load(file)

                file_data.append(r.json()["Time Series (1min)"])
        
                file.seek(0)
        
                json.dump(file_data, file)
        month += 1
        print("waiting 10 secs...")
        time.sleep(5)
        print("avoided rate limit")
    month = 0
    year += 1    
    

#url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&month=2019-11&outputsize=full&adjusted=false&datatype=json&extended_hours=false&apikey=3BAAT9BW5TUZBZY8'
#r = requests.get(url)
#data = r.json()

#print(data)
#print("done")