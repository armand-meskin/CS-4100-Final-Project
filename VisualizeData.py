import json
import matplotlib.pyplot as plt

def load_dat(filename):
    f = open(filename)
    dat = json.load(f)
    arr_dat = [] 
    for sub_dict in dat.values():
        arr_dat.append(float(sub_dict['4. close']))
    return arr_dat



X_Train = load_dat('data.json')
iteration = []
for i in range(0, len(X_Train)):
    iteration.append(i)

# X and Y are reversed here
plt.plot(iteration, X_Train, linewidth=0.75)
#plt.axis((0, 600000, 0, 500))
plt.ylabel('Price in USD')
plt.xlabel('Minutes')
plt.show()