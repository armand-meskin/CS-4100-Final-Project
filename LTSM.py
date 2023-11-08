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

# We need to take out all chains of 4 weeks that contain holidays
def process_dat(data):
    date_format = '%Y-%m-%d %H:%M:%S'
    proccessed_dat = dict()
    temp = []
    indicator = 0 # Monday
    chain_len = 0
    for key in data.keys():
        day = datetime.strptime(key, date_format)
        #print(str(day.weekday()))
        if day.weekday() == indicator:
            temp.append(key)
        else:
            if day.weekday() == indicator + 1:
                temp.append(key)
                indicator += 1
                continue
            elif indicator == 4:
                indicator = 0
                chain_len += 1
                # temp contains keys for a valid week
                if chain_len == 4:
                    chain_len = 0
                    for key in temp:
                        proccessed_dat[key] = data[key]
                    temp = []
            else:
                # This week is incomplete
                indicator = 0
                chain_len = 0
                temp = []
    
            
    
    with open('processed-data.json', 'w') as file:
        json.dump(proccessed_dat, file, indent=4)




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