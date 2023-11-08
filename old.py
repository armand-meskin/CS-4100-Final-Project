# We need to take out all chains of 4 weeks that contain holidays
def process_dat(data):
    date_format = '%Y-%m-%d %H:%M:%S'
    proccessed_dat = dict()
    temp = []
    indicator = 0 # Monday
    chain_len = 0
    skip = False
    for key in data.keys():
        day = datetime.strptime(key, date_format)

        # Alpha vanatage has damaged data for this day
        if '2021-04-12' in key:
            indicator = 0
            temp = []
            chain_len = 0
            skip = True

        # Skips remaining mins if monday holiday
        if skip:
            if day.weekday() == 1:
                continue
            else:
                skip = False

        #print(str(day.weekday()))
        if day.weekday() == indicator:
            temp.append(key)
        else:
            if day.weekday() == indicator + 1:
                #print(f"Current key: {key}")
                temp.append(key)
                indicator += 1
            elif indicator == 4:
                indicator = 0
                chain_len += 1
                # temp contains keys for a valid week
                if chain_len == 4:
                    chain_len = 0
                    for key2 in temp:
                        proccessed_dat[key2] = data[key2]
                    temp = []
                if day.weekday() == 1:
                    chain_len = 0
                    temp = []
                    skip = True
                    indicator = -1
                elif day.weekday() == indicator:
                    temp.append(key)

            elif indicator == 3 and day.weekday() == 0:
                indicator = 0
                if day.weekday() == indicator:
                    temp.append(key)
            else:
                # This week is incomplete
                #print(f"Current key: {key}")
                indicator = 0
                chain_len = 0
                temp = []
    
            
    print(len(proccessed_dat.keys()))

    for key in proccessed_dat.keys():
        date, time = key.split(" ")
        date = date + " 09:30:00"
        #print(date)
        if date not in proccessed_dat.keys():
           print(date)
            

    
    with open('processed-data.json', 'w') as file:
        json.dump(proccessed_dat, file, indent=4)