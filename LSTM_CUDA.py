from DataProcessing import load_dat_array
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Make training data in increments of m_delta, m_delta must be a divisor of 7800 (four weeks of mins)
def make_traing(raw, m_delta):
    print("Prepping data, this may take a minute.")
    X_Train = []
    Y_Train = []

    temp = []
    idx = 0
    offset = 0
    # Implement sliding window
    while True:
        if idx + (m_delta*2) + offset >= len(raw):
            # There is no more data to add
            print(f"Finished at idx: {idx} and off: {offset}")
            break

        temp.append(raw[idx + offset])
        idx += m_delta

        # Four weeks passed make an example
        if idx % 7800 == 0:
            #print(f'idx is {idx} and off is {offset}')
            temp.append(raw[idx + offset])
            X_Train.append(temp.copy())
            Y_Train.append([raw[idx + offset + m_delta]])
            temp = []
            idx = 0
            offset += 1

    diff = len(X_Train) - round(len(X_Train) * 0.9)
    X_Valid = []
    Y_Valid = []
    for i in range(diff):
        X_Valid.append(X_Train.pop())
        Y_Valid.append(Y_Train.pop())
    X_Valid.reverse()
    Y_Valid.reverse()

    return X_Train, Y_Train, X_Valid, Y_Valid
        


class Net(nn.Module):
    def __init__(self, in_s, out_s):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=in_s, hidden_size=128, 
                            num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(in_features=128, out_features=out_s)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden

raw_data = load_dat_array('processed-data.json')

delta = 60
X_Train, Y_Train, X_Valid, Y_Valid = make_traing(raw_data, delta)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

X_Train = torch.tensor(X_Train, dtype=torch.float).to(device)
Y_Train = torch.tensor(Y_Train, dtype=torch.float).to(device)
X_Valid = torch.tensor(X_Valid, dtype=torch.float).to(device)
Y_Valid = torch.tensor(Y_Valid, dtype=torch.float).to(device)

#lstm = nn.LSTM(7800/delta, 3)
in_size = len(X_Train[0])
out_size = len(Y_Train[0])

print("Hidden: ", in_size)

model = Net(in_size, out_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

valid_loss_min = float("inf")
bad_epoch = 0
global_step = 0
for epoch in tqdm(range(15000)):
    model.train()
    train_loss_array = []
    hidden_train = None

    optimizer.zero_grad()
    pred_Y, hidden_train = model(X_Train, hidden_train)
    hidden_train = None

    loss = criterion(pred_Y, Y_Train)  
    loss.backward()               
    optimizer.step()                  
    train_loss_array.append(loss.item())
    global_step += 1

    #model.eval()
    #valid_loss_array = []
    #hidden_valid = None

    #for idx in range(len(X_Valid)):
    #    pred_Y, hidden_valid = model(X_Valid, hidden_valid)
    #    hidden_valid = None
    #    loss = criterion(pred_Y, Y_Valid[idx])
    #    valid_loss_array.append(loss.item())

    train_loss_cur = np.mean(train_loss_array)
    #valid_loss_cur = np.mean(valid_loss_array)
    print(f"The train loss is {train_loss_cur} and the valid loss is N/A.")

    #if valid_loss_cur < valid_loss_min:
    #        valid_loss_min = valid_loss_cur
    #        bad_epoch = 0
    #        torch.save(model.state_dict(), 'model-save')
    #else:
    #    bad_epoch += 1
    #    if bad_epoch >= 5:
    #        print(f"The training stops early in epoch {epoch}")
    #        break

valid_loss_array = []

model.eval()
hidden_valid = None

pred_Y, hidden_valid = model(X_Valid, hidden_valid)
loss = criterion(pred_Y, Y_Valid)
valid_loss_array.append(loss.item())

valid_loss_cur = np.mean(valid_loss_array)
print(valid_loss_cur)
print(pred_Y)