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
    print("Preping data, this may take a minute.")
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
        


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[-1])
        return predictions

raw_data = load_dat_array('processed-data.json')

delta = 60
X_Train, Y_Train, X_Valid, Y_Valid = make_traing(raw_data, delta)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

# Example input and output dimensions
input_size = 1  # Number of input features (in this case, 1 feature at each time step)
hidden_size = 20  # Number of features in the hidden state
output_size = 1   # Number of output features (single value prediction)

model = SimpleLSTM(input_size, hidden_size, output_size)

x_train = torch.tensor(X_Train, dtype=torch.float32)
y_train = torch.tensor(Y_Train, dtype=torch.float32)

x_val = torch.tensor(X_Valid, dtype=torch.float32)
y_val = torch.tensor(Y_Valid, dtype=torch.float32)
print("Stuck?")

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Stuck?2")
def train(model, loss_function, optimizer, x_train, y_train, x_val, y_val, epochs=50):
    print("Stuck?3")
    for epoch in range(epochs):
        print("Stuck?4")
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_train)

        # Compute loss
        loss = loss_function(y_pred, y_train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = loss_function(y_val_pred, y_val)
            print(f"Validation Loss: {val_loss.item()}")

# Train the model
train(model, loss_function, optimizer, x_train, y_train, x_val, y_val)



