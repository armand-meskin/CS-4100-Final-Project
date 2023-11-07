import requests
import json
import time
import os
# Collect the past 5 years and store in csv...
# End at 2023-11 (yr-month)
# Start at 2018-11 (yr-month)
year = 2018
month = 11
runningDict = dict()
    

while True:
    while month != 13:
        formatMonth = month
        if formatMonth < 10:
            formatMonth = "0" + str(formatMonth)
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&month={year}-{formatMonth}&outputsize=full&adjusted=false&datatype=json&extended_hours=false&apikey=3BAAT9BW5TUZBZY8'
        r = requests.get(url)
        print("Request revieved...")
        for key in r.json()["Time Series (1min)"]:
            runningDict[key] = r.json()["Time Series (1min)"][key]
        print(f"Completed entries for req for: {year} - {month}")
        month += 1
        if year == 2023 and month == 12:
            print("Five years accumulated...")
            break
        print("waiting 4 secs...")
        time.sleep(4)
        print("avoided rate limit")
    if year == 2023 and month == 12:
        break
    month = 0
    year += 1

with open('data.json', 'w') as file:
    json.dump(runningDict, file, indent=4)
#url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&month=2019-11&outputsize=full&adjusted=false&datatype=json&extended_hours=false&apikey=3BAAT9BW5TUZBZY8'
#r = requests.get(url)
#data = r.json()

#print(data)
print("done")