import json
from datetime import datetime

# Load data from json to an array
def load_dat_array(filename):
    f = open(filename)
    dat = json.load(f)
    arr_dat = [] 
    for sub_dict in dat.values():
        arr_dat.append(float(sub_dict['4. close']))
    return arr_dat

# Load data from json to a dictionary
def load_dat_dict(filename):
    f = open(filename)
    return json.load(f)



# Our data needs to be in chains of four weeks and we need to interpolate fake holiday data
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