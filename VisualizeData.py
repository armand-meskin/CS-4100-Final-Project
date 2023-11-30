from DataProcessing import load_dat_array
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

def sliding_window(raw, m_delta):
    X = []
    y = []

    temp = []
    idx = 0
    offset = 0
    # Implement sliding window
    while True:
        if idx + (m_delta * 2) + offset >= len(raw):
            # There is no more data to add
            print(f"Finished at idx: {idx} and off: {offset}")
            break

        temp.append(raw[idx + offset])
        idx += m_delta

        # Four weeks passed make an example
        if idx % 7800 == 0:
            temp.append(raw[idx + offset])
            X.append(temp.copy())
            y.append([raw[idx + offset + m_delta]])
            temp = []
            idx = 0
            offset += 1
    return np.array(X), np.array(y)

def movement_indicator(actual, predictions, div):
    # Build tuples
    correct = 0
    for i in range(len(actual)-60):
        temp_a = ''
        temp_p = ''
        if actual[i] < actual[i+60]:
            temp_a = 'up'
        else:
            temp_a = 'down'

        if predictions[i] < predictions[i+60]:
            temp_p = 'up'
        else:
            temp_p = 'down'
        
        if i >= div and temp_a == temp_p:
            correct += 1

    return correct / (len(actual) - div)

        

print("Preparing data...")

raw_data = load_dat_array('processed-data.json')

X, y = sliding_window(raw_data, 60)

ss = StandardScaler()

X = ss.fit_transform(X)
y = ss.fit_transform(y.reshape(-1, 1))

div = int(len(X) * 0.8)

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = F.relu(hn.view(-1, self.hidden_size))
        out = self.fc(hn)
        return out


input_size = 131
hidden_size = 131
num_layers = 1

model = CustomLSTM(input_size, hidden_size, num_layers)

model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()
to_predict = torch.Tensor(X)
to_predict = torch.reshape(to_predict, shape=(to_predict.shape[0], 1, to_predict.shape[1]))

train_predict = model(to_predict)
pred = train_predict.data.numpy()

pred = ss.inverse_transform(pred)
y = ss.inverse_transform(y)

print(movement_indicator(y, pred, div))
plt.axvline(x=div, c='r', linestyle='--')

plt.plot(y, label="Actual")
plt.plot(pred, label="Prediction")
plt.show()


