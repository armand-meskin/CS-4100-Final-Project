from DataProcessing import load_dat_array

def make_traing(raw):
    X_Train = []
    Y_Train = []

    temp = []
    idx = 0
    offset = 0
    while True:
        if idx + 120 + offset >= len(raw):
            # There is no more data to add
            print(f"broke at idx: {idx} and off: {offset}")
            break

        temp.append(raw[idx + offset])
        idx += 60

        # Four weeks passed make an example
        if idx % 7800 == 0:
            print(f'idx is {idx} and off is {offset}')
            temp.append(raw[idx + offset])
            X_Train.append(temp.copy())
            Y_Train.append([raw[idx + offset + 60]])
            temp = []
            idx = 0
            offset += 1
    return X_Train, Y_Train
        



raw_data = load_dat_array('processed-data.json')

X_Train, Y_Train = make_traing(raw_data)

print(X_Train[len(X_Train) - 1])
