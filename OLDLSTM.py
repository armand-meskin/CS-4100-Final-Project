from DataProcessing import load_dat_array
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def sliding_window(raw, dates, m_delta):
    X = []
    y = []
    d = []

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
            d.append([datetime.strptime(dates[idx + offset + m_delta], '%Y-%m-%d %H:%M:%S')])
            temp = []
            idx = 0
            offset += 1
    return np.array(X), np.array(y), np.array(d)

print("Preparing data...")

keys, raw_data = load_dat_array('processed-data.json')

X, y, x_dates = sliding_window(raw_data, list(keys), 60)

ss = StandardScaler()

X = ss.fit_transform(X)
y = ss.fit_transform(y.reshape(-1, 1))

div = int(len(X) * 0.8)

X_train = X[:div]
X_test = X[div:]

y_train = y[:div]
y_test = y[div:]

X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)

y_train_tensor = torch.Tensor(y_train)
y_test_tensor = torch.Tensor(y_test)

X_train_tensor = torch.reshape(X_train_tensor, (X_train_tensor.shape[0], 1, X_train_tensor.shape[1]))
X_test_tensor = torch.reshape(X_test_tensor, (X_test_tensor.shape[0], 1, X_test_tensor.shape[1]))

X_train_tensor.requires_grad_(True)

print("Data ready.")


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


# Parameters
input_size = X_train_tensor.shape[2]
hidden_size = 131
num_layers = 1
num_epochs = 20
learning_rate = 0.05

model = CustomLSTM(input_size, hidden_size, num_layers)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
grads = []

print("Beginning training...")
for epoch in tqdm(range(num_epochs)):
    outputs = model.forward(X_train_tensor)
    optimizer.zero_grad()

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    grads.append(X_train_tensor.data.grad)
    if epoch % (num_epochs // 10) == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

print("Finished training.")
torch.save(model.state_dict(), 'model_state_dict.pth')

