import json
from datetime import datetime, timedelta

# Load data from json to an array
def load_dat_array(filename):
    f = open(filename)
    dat = json.load(f)
    f.close()
    arr_dat = [] 
    for sub_dict in dat.values():
        arr_dat.append(float(sub_dict['4. close']))
    return arr_dat

# Load data from json to a dictionary
def load_dat_dict(filename): 
    f = open(filename)
    dat = json.load(f)
    f.close()
    return dat



# Our data needs to be in chains of four weeks and we need to interpolate fake holiday and damaged data
def process_dat(data):
    date_format = '%Y-%m-%d %H:%M:%S'
    data_truncate = data.copy()

    # Omit the first days in our data that arent Monday
    for key in data.keys():
        day = datetime.strptime(key, date_format)
        if day.weekday() != 0:
            data_truncate.pop(key)
        else:
            break


    # Fix damaged data that is missing minutes
    data_array = []
    for id in data_truncate.keys():
        data_array.append((id, data_truncate[id]))

    index = 1
    cleaned = False

    while not cleaned:
        cleaned = True
        print("Beginning new runthrough")
        for key in data_truncate.keys():
            if index == len(data_truncate.keys()):
                # Finished one passthrough
                index = 1
                break
            day = datetime.strptime(key, date_format)
            dayNext = day + timedelta(minutes=1)
            if dayNext.strftime(date_format) == data_array[index][0] or '15:59:00' in key:
                # All is well or we hit the end of a week
                index += 1
            else:
                # Damaged data found, interpolate missing point
                print(f"Damaged point found: {dayNext.strftime(date_format)} when comparing {data_array[index][0]}")
                cleaned = False
                open2 = float(data_truncate[key]['1. open']) + float(data_array[index][1]['1. open'])
                high = float(data_truncate[key]['2. high']) + float(data_array[index][1]['2. high'])
                low = float(data_truncate[key]['3. low']) + float(data_array[index][1]['3. low'])
                close = float(data_truncate[key]['4. close']) + float(data_array[index][1]['4. close'])
                volume = float(data_truncate[key]['5. volume']) + float(data_array[index][1]['5. volume'])

                temp = dict()
                temp['1. open'] = str(round(open2/2, 4))
                temp['2. high'] = str(round(high/2, 4))
                temp['3. low'] = str(round(low/2, 4))
                temp['4. close'] = str(round(close/2, 4))
                temp['5. volume'] = str(round(volume/2, 4))

                # Insert fabricated point
                data_array.insert(index, (dayNext.strftime(date_format), temp.copy()))
                index += 2

        # Update our main dict
        print("Updating main dict")
        data_truncate = dict(data_array)

    #out = open("test-file.json", "w") 
    #json.dump(data_truncate, out, indent=4)

    
    indicator = 0
    holidays = []
    for key in data_truncate.keys():
        day = datetime.strptime(key, date_format)
        # Day is as expected
        if day.weekday() == indicator:
            # Closed stock market on weekends wrap to Monday
            if indicator == 4:
                indicator = 0
            else:
                indicator += 1
        # Day is unexpected there is a Holiday
        else:
            holidays.append(key)
            print("Hi")



# Given a holiday interpolate data between the next open market day and the previous
def interpolate():
    print("Placeholder")





# Format array of data into an array of examples
def build_examples(data):
    # Offset begins at 0
    offset = 0
    # Thirty min increments in an example
    spacing = 30
    # 390 in a day so 30*390 is a month
    example_space = 11700


data = load_dat_dict('data.json')
process_dat(data)