import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device('cpu')

print(device)

print('Preparing data...')

data_df = pd.read_csv('pepsi_data.csv', index_col='index')

y = np.array(data_df['target'])
X = data_df.drop(['target', 'time'], axis=1).to_numpy()

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

print('Data ready.')

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
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) # internal state
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = F.relu(hn.view(-1, self.hidden_size))
        out = self.fc(hn)
        return out


print('Beginning Training...')
# parameters
input_size = X_train_tensor.shape[2]
hidden_size = 130
num_layers = 1
epochs = 100

model = CustomLSTM(input_size, hidden_size, num_layers, X_train_tensor.shape[1])
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

for epoch in tqdm(range(epochs)):
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    outputs = model.forward(X_train_tensor)
    optimizer.zero_grad()

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % (epochs // 10) == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

print('Finished Training, beginning validation...')
torch.save(model.state_dict(), 'pepsi_state_dict.pth')
