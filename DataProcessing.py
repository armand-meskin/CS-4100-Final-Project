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
    '''
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
                if '15:59:00' in key and '09:30:00' not in data_array[index][0]:
                    # First  point of next days data is damaged
                    print("Prob point at", key)
                    print("idx", data_array[index][0])
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
                    wrap = day + timedelta(days=1)
                    if wrap.weekday() == 5:
                        wrap = wrap + timedelta(days=2)
                    wrap = wrap.strftime('%Y-%m-%d').split(" ")[0] + ' 09:30:00'
                    data_array.insert(index, (wrap, temp.copy()))
                    index += 1
                
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
    '''
    # THIS IS TEMPORARY
    data_truncate = load_dat_dict('test-file.json')
    data_array = []
    for id in data_truncate.keys():
        data_array.append((id, data_truncate[id]))



    index = 390
    cmp = " "
    for key in data_truncate.keys():
        #print(f"First pass {key} and first idx {data_array[0][0]}")
        if index >= len(data_array):
            break
        temp, extra = key.split(" ")

        if cmp == temp:
            # This date has already been taken care of skip
            continue
        else:
            cmp, extra = key.split(" ")

        day = datetime.strptime(key, date_format)
        dayNext = day + timedelta(days=1)

        if dayNext.weekday() < 5:
            # Is a weekday
            if dayNext.strftime('%Y-%m-%d') == data_array[index][0].split(" ")[0]:
                #print("The following day is present")
                index += 390
            else:
                # The following day is not present, there is a holiday
                # interploate...
                #print(f"Failed {dayNext.strftime('%Y-%m-%d')} and {data_array[index][0]} idx is {index}")
                date2, time2 = data_array[index][0].split(" ")
                generated_day = interpolate(dayNext.strftime('%Y-%m-%d'), day.strftime('%Y-%m-%d'), date2, data_truncate)
                print("Inserted Holiday at Weekday: ", dayNext.strftime('%Y-%m-%d'))
                print("mismatched idx: ", data_array[index][0])
                for item in generated_day:
                    data_array.insert(index, item)
                    index += 1
                index += 390
                print("idx after: ", data_array[index][0])

        else:
            # The next day is Saturday everything is normal so far all we need to do is check that monday follows in our data_array
            mon = dayNext + timedelta(days=2)
            if mon.weekday() != 0:
                raise Exception(f"mon is not Monday found: {mon.weekday()} instead")
            if mon.strftime('%Y-%m-%d') == data_array[index][0].split(" ")[0]:
                index += 390
                continue
            else:
                # Monday is a holiday interpolate...
                #print(f"Failed {mon.strftime('%Y-%m-%d')} and {data_array[index][0]} idx is {index}")
                date2, time2 = data_array[index][0].split(" ")
                generated_day = interpolate(mon.strftime('%Y-%m-%d'), day.strftime('%Y-%m-%d'), date2, data_truncate)
                print("Inserted Holiday at Monday: ", mon.strftime('%Y-%m-%d'))
                print("mismatched idx: ", data_array[index][0])
                for item in generated_day:
                    data_array.insert(index, item)
                    index += 1
                index += 390

    out = open("test-file2.json", "w") 
    json.dump(dict(data_array), out, indent=4)



# Given a holiday interpolate data between the next open market day and the previous
def interpolate(holiday, previous_day, next_day, data_trunc):
    date_format = '%Y-%m-%d %H:%M:%S'
    start_key_prev = datetime.strptime(previous_day + ' 09:30:00', date_format)
    start_key_next = datetime.strptime(next_day + ' 09:30:00', date_format)
    holiday = datetime.strptime(holiday + ' 09:30:00', date_format)
    prev_mins = []
    next_mins = []

    generated = []
    while True:
        open2 = float(data_trunc[start_key_next.strftime(date_format)]['1. open']) + float(data_trunc[start_key_prev.strftime(date_format)]['1. open'])
        high = float(data_trunc[start_key_next.strftime(date_format)]['2. high']) + float(data_trunc[start_key_prev.strftime(date_format)]['2. high'])
        low = float(data_trunc[start_key_next.strftime(date_format)]['3. low']) + float(data_trunc[start_key_prev.strftime(date_format)]['3. low'])
        close = float(data_trunc[start_key_next.strftime(date_format)]['4. close']) + float(data_trunc[start_key_prev.strftime(date_format)]['4. close'])
        volume = float(data_trunc[start_key_next.strftime(date_format)]['5. volume']) + float(data_trunc[start_key_prev.strftime(date_format)]['5. volume'])

        temp = dict()

        temp['1. open'] = str(round(open2/2, 4))
        temp['2. high'] = str(round(high/2, 4))
        temp['3. low'] = str(round(low/2, 4))
        temp['4. close'] = str(round(close/2, 4))
        temp['5. volume'] = str(round(volume/2, 4))

        generated.append((holiday.strftime(date_format), temp.copy()))
        # Last data point to be generated
        if '15:59:00' in holiday.strftime(date_format):
            return generated
        start_key_prev = start_key_prev + timedelta(minutes=1)
        start_key_next = start_key_next + timedelta(minutes=1)
        holiday = holiday + timedelta(minutes=1)
        







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

#print(interpolate('2021-07-29', '2021-07-28', '2018-11-30', data))