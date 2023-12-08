import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import numpy as np

STOCK_NAME = 'pepsi'

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = F.relu(hn.view(-1, self.hidden_size))
        out = self.fc(hn)
        return out


def movement_indicator(actual, predictions, div):
    # Build tuples
    correct = 0
    for i in range(len(actual) - 60):
        if actual[i] < actual[i + 60]:
            temp_a = 'up'
        else:
            temp_a = 'down'

        if predictions[i] < predictions[i + 60]:
            temp_p = 'up'
        else:
            temp_p = 'down'

        if i >= len(actual) * div and temp_a == temp_p:
            correct += 1

    return correct / (len(actual) - int(len(actual) * div))


data_df = pd.read_csv(f'all_data/{STOCK_NAME}_data.csv')

X = data_df.drop(['target', 'time'], axis=1).to_numpy()
X = np.delete(X, len(X[0]) - 1, 1)
y = np.array(data_df['target'])
d = pd.to_datetime(data_df['time'])

ss = StandardScaler()

X = ss.fit_transform(X)
y = ss.fit_transform(y.reshape(-1, 1))

div = 0.8

to_predict = torch.Tensor(X)
to_predict = torch.reshape(to_predict, shape=(to_predict.shape[0], 1, to_predict.shape[1]))

input_size = to_predict.shape[2]
hidden_size = 130
num_layers = 1

model = CustomLSTM(input_size, hidden_size, num_layers, to_predict.shape[1])
model.load_state_dict(torch.load(f'{STOCK_NAME}_state_dict.pth'))

train_predict = model(to_predict)
pred = train_predict.data.numpy()

pred = ss.inverse_transform(pred)
y = ss.inverse_transform(y)

print(movement_indicator(y, pred, div))

plt.axvline(x=d[int(div * len(y))], color='r', linestyle='dashed')
plt.plot(d, y, label="Actual")
plt.plot(d, pred, label="Prediction")
plt.xlabel('Date')
plt.ylabel('Stock Closing Price')
plt.title(f'LSTM Predicts {STOCK_NAME.capitalize()} Stock')
plt.legend()
plt.show()



