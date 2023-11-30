import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def validate_model(model, X_val, Y_val, criterion, num_layers, hidden_size):
    model.eval()  # Set the model to evaluation mode

    hidden = (torch.zeros(num_layers, 1, hidden_size),
              torch.zeros(num_layers, 1, hidden_size))

    val_loss = 0.0
    results = []
    with torch.no_grad():  # Turn off gradients
        for i in tqdm(range(len(X_val))):
            hidden = tuple([h.detach() for h in hidden])
            # Forward pass
            #print(X_val[i].unsqueeze(1))
            out, hidden = model(X_val[i].unsqueeze(1), hidden)
            results.append(out)
            # Compute Loss
            loss = criterion(out[-1], Y_val[i])
            val_loss += loss.item()
    #print("Actual", Y_val[len(Y_val) - 1])
    print("Predicted", results)
    return val_loss / len(X_val)

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_rate=0.0):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[-1])
        return out, hidden

input_size = 3
num_layers = 1
hidden_size = 2

X_Train = torch.tensor([[1,2,3], [2,3,4], [3,4,5]], dtype=torch.float).view(-1, 1, input_size)

Y_Train = torch.tensor([[1], [1.5], [2]], dtype=torch.float)

X_Valid = torch.tensor([[4,5,6], [5, 6, 7], [6, 7, 8]], dtype=torch.float).view(-1, 1, input_size)

Y_Valid = torch.tensor([[2.5], [3], [3.5]], dtype=torch.float)


model = CustomLSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout_rate=0)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10
hidden = (torch.zeros(num_layers, 1, hidden_size),
              torch.zeros(num_layers, 1, hidden_size))

for epoch in range(num_epochs):
    model.train()

    average_loss = []
    for i in tqdm(range(len(X_Train))):
        # Detach the hidden state to prevent in-place modifications
        hidden = tuple([h.detach() for h in hidden])

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        out, hidden = model(X_Train[i].unsqueeze(1), hidden)
        # Compute Loss
        loss = criterion(out[-1], Y_Train[i])
        average_loss.append(loss.item())
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print("AVG: ", sum(average_loss) / len(average_loss))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

val_loss = validate_model(model, X_Valid, Y_Valid, criterion, num_layers, hidden_size)
print(f'Validation Loss: {val_loss}')