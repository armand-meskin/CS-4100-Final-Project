import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

STOCK_NAME = 'nvidia'


# Load data from pandas to a dictionary
def load_dat_dict(filename):
    data_df = pd.read_csv(filename, index_col='time').drop('index', axis=1)
    dat = data_df.to_dict(orient='index')
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
                if '15:59:00' in key and '09:30:00' not in data_array[index][0]:
                    # First  point of next days data is damaged
                    print("Prob point at", key)
                    print("idx", data_array[index][0])
                    cleaned = False

                    open2 = float(data_truncate[key]['open']) + float(data_array[index][1]['open'])
                    high = float(data_truncate[key]['high']) + float(data_array[index][1]['high'])
                    low = float(data_truncate[key]['low']) + float(data_array[index][1]['low'])
                    close = float(data_truncate[key]['close']) + float(data_array[index][1]['close'])
                    volume = float(data_truncate[key]['volume']) + float(data_array[index][1]['volume'])

                    temp = dict()
                    temp['open'] = str(round(open2 / 2, 4))
                    temp['high'] = str(round(high / 2, 4))
                    temp['low'] = str(round(low / 2, 4))
                    temp['close'] = str(round(close / 2, 4))
                    temp['volume'] = str(round(volume / 2, 4))

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
                open2 = float(data_truncate[key]['open']) + float(data_array[index][1]['open'])
                high = float(data_truncate[key]['high']) + float(data_array[index][1]['high'])
                low = float(data_truncate[key]['low']) + float(data_array[index][1]['low'])
                close = float(data_truncate[key]['close']) + float(data_array[index][1]['close'])
                volume = float(data_truncate[key]['volume']) + float(data_array[index][1]['volume'])

                temp = dict()
                temp['open'] = str(round(open2 / 2, 4))
                temp['high'] = str(round(high / 2, 4))
                temp['low'] = str(round(low / 2, 4))
                temp['close'] = str(round(close / 2, 4))
                temp['volume'] = str(round(volume / 2, 4))

                # Insert fabricated point
                data_array.insert(index, (dayNext.strftime(date_format), temp.copy()))
                index += 2

        # Update our main dict
        print("Updating main dict")
        data_truncate = dict(data_array)
    index = 390
    cmp = " "
    for key in data_truncate.keys():
        # print(f"First pass {key} and first idx {data_array[0][0]}")
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
                # print("The following day is present")
                index += 390
            else:
                # The following day is not present, there is a holiday
                # interploate...
                # print(f"Failed {dayNext.strftime('%Y-%m-%d')} and {data_array[index][0]} idx is {index}")
                date2, time2 = data_array[index][0].split(" ")
                generated_day = interpolate(dayNext.strftime('%Y-%m-%d'), day.strftime('%Y-%m-%d'), date2,
                                            data_truncate)
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
                # print(f"Failed {mon.strftime('%Y-%m-%d')} and {data_array[index][0]} idx is {index}")
                date2, time2 = data_array[index][0].split(" ")
                generated_day = interpolate(mon.strftime('%Y-%m-%d'), day.strftime('%Y-%m-%d'), date2, data_truncate)
                print("Inserted Holiday at Monday: ", mon.strftime('%Y-%m-%d'))
                print("mismatched idx: ", data_array[index][0])
                for item in generated_day:
                    data_array.insert(index, item)
                    index += 1
                index += 390

    # Remove trailing days at end of our data from incomplete weeks...
    while True:
        if datetime.strptime(data_array[len(data_array) - 1][0], date_format).weekday() != 4:
            data_array.pop()
        else:
            break

    # Ensure our data is in chains of four weeks
    while True:
        # 1950 points is one week worth of points
        if (len(data_array) / 1950) % 4 != 0:
            for i in range(1950):
                # Remove the week thats the least recent 
                data_array.pop(0)
        else:
            print("Data is in chains of four")
            break

    verify_itegrity(dict(data_array))
    out = open("processed_data.json", "w")
    json.dump(dict(data_array), out, indent=4)
    return dict(data_array)


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
        open2 = float(data_trunc[start_key_next.strftime(date_format)]['open']) + float(
            data_trunc[start_key_prev.strftime(date_format)]['open'])
        high = float(data_trunc[start_key_next.strftime(date_format)]['high']) + float(
            data_trunc[start_key_prev.strftime(date_format)]['high'])
        low = float(data_trunc[start_key_next.strftime(date_format)]['low']) + float(
            data_trunc[start_key_prev.strftime(date_format)]['low'])
        close = float(data_trunc[start_key_next.strftime(date_format)]['close']) + float(
            data_trunc[start_key_prev.strftime(date_format)]['close'])
        volume = float(data_trunc[start_key_next.strftime(date_format)]['volume']) + float(
            data_trunc[start_key_prev.strftime(date_format)]['volume'])

        temp = dict()

        temp['open'] = str(round(open2 / 2, 4))
        temp['high'] = str(round(high / 2, 4))
        temp['low'] = str(round(low / 2, 4))
        temp['close'] = str(round(close / 2, 4))
        temp['volume'] = str(round(volume / 2, 4))

        generated.append((holiday.strftime(date_format), temp.copy()))
        # Last data point to be generated
        if '15:59:00' in holiday.strftime(date_format):
            return generated
        start_key_prev = start_key_prev + timedelta(minutes=1)
        start_key_next = start_key_next + timedelta(minutes=1)
        holiday = holiday + timedelta(minutes=1)


# Verify data format
def verify_itegrity(data):
    data_array = []
    for id in data.keys():
        data_array.append((id, data[id]))
    date_format = '%Y-%m-%d %H:%M:%S'
    next = datetime.strptime(data_array[0][0], date_format)
    count = 0
    for item in data_array:
        current = datetime.strptime(item[0], date_format)
        if item[0] != next.strftime(date_format):
            print("Integrity violation at item: ", item)
            print("Expected: ", next)
            raise Exception("Integrity Violation")

        if "15:59:00" in item[0] and current.weekday() == 4:
            next = current + timedelta(days=3)
            temp = next.strftime(date_format).split(" ")[0] + " 09:30:00"
            next = datetime.strptime(temp, date_format)
        elif "15:59:00" in item[0]:
            next = current + timedelta(days=1)
            temp = next.strftime(date_format).split(" ")[0] + " 09:30:00"
            next = datetime.strptime(temp, date_format)
        else:
            next = current + timedelta(minutes=1)
        count += 1
    print(f"Integrity verfied, {len(data_array)} verfied points")


data = load_dat_dict(f'all_data/{STOCK_NAME}_data.csv')
processed = process_dat(data)

processed = load_dat_dict('test-file2.json')
verify_itegrity(processed)

# we used json originally, but switched to pandas. so now we convert back to csv format
def sliding_window(f_name, chunk_size, m_delta):
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


data = json.load(open('processed_data.json'))
raw, dates = np.array([i['4. close'] for i in data.values()]), np.array(list(data.keys()))

X, y, d = sliding_window(raw, dates, 7800, 60)

final_dict = {'time': d, 'target': y}

for i in tqdm(range(len(X))):
    for j in range(len(X[i])):
        if i == 0:
            final_dict[f'feature {j}'] = [X[i][j]]
        else:
            final_dict[f'feature {j}'].append(X[i][j])

pd.DataFrame(final_dict).to_csv(f'all_data/{STOCK_NAME}_data.csv', index_label='index')
