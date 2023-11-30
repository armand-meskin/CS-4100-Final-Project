from DataProcessing import load_dat_array
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
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

print("Preparing data...")

raw_data = load_dat_array('processed-data.json')

X, y = sliding_window(raw_data, 60)

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

print("Data ready.")


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.fc(hn)
        return out


# Parameters
input_size = X_train_tensor.shape[2]
hidden_size = 2
num_layers = 1
num_epochs = 100
learning_rate = 0.1

model = CustomLSTM(input_size, hidden_size, num_layers)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Beginning training...")
for epoch in tqdm(range(num_epochs)):
    outputs = model.forward(X_train_tensor)
    optimizer.zero_grad()

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % (num_epochs // 10) == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

print("Finished training.")

to_predict = torch.Tensor(X)
to_predict = torch.reshape(to_predict, shape=(to_predict.shape[0], 1, to_predict.shape[1]))

train_predict = model(to_predict)
pred = train_predict.data.numpy()

pred = ss.inverse_transform(pred)
y = ss.inverse_transform(y)

plt.axvline(x=div, c='r', linestyle='--')

plt.plot(y, label="Actual")
plt.plot(pred, label="Prediction")
plt.show()
