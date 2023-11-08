import json

# Load data from json
def load_dat(filename):
    f = open(filename)
    dat = json.load(f)
    arr_dat = [] 
    for sub_dict in dat.values():
        arr_dat.append(float(sub_dict['4. close']))
    return arr_dat

# Format array of data into an array of examples
def build_examples(data):
    # Offset begins at 0
    offset = 0
    # One hour increments in an example
    spacing = 60
    # 390 in a day so 30*390 is a month
    example_space = 11700