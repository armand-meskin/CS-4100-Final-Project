from DataProcessing import load_dat_array
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    print("Predicted", results[1])
    print("Predicted", results[len(results) - 1])
    return val_loss / len(X_val)
        

raw_data = load_dat_array('processed-data.json')


delta = 60
X_Train, Y_Train, X_Valid, Y_Valid = make_traing(raw_data, delta)

print(len(X_Train))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is: {device}")

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop, output_size=1):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[-1])
        out = self.relu(out)
        return out, hidden

# Parameters
input_size = len(X_Train[0])
hidden_size = 4  # Adjust as needed
num_layers = 1
num_epochs = 1  # Increase as needed
learning_rate = 0.1  # Adjust as needed
dropout_rate = 0.9  # Optional, add if overfitting is observed

model = CustomLSTM(input_size, hidden_size, num_layers, dropout_rate)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


XTrain_tensor = torch.tensor(X_Train, dtype=torch.float, requires_grad=True).view(-1, 1, input_size)
YTrain_tensor = torch.tensor(Y_Train, dtype=torch.float)

XValid_tensor = torch.tensor(X_Valid, dtype=torch.float, requires_grad=False).view(-1, 1, input_size)
YValid_tensor = torch.tensor(Y_Valid, dtype=torch.float)

XTrain_tensor.requires_grad_(True)

#min_X = XTrain_tensor.min()
#max_X = XTrain_tensor.max()
#min_Y = YTrain_tensor.min()
#max_Y = YTrain_tensor.max()

#XTrain_tensor = (XTrain_tensor - min_X) / (max_X - min_X)
#YTrain_tensor = (YTrain_tensor - min_Y) / (max_Y - min_Y)

#XValid_tensor = (XValid_tensor - min_X) / (max_X - min_X)
#YValid_tensor = (YValid_tensor - min_Y) / (max_Y - min_Y)

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
        out, hidden = model(XTrain_tensor[i].unsqueeze(1), hidden)
        # Compute Loss
        loss = criterion(out[-1], YTrain_tensor[i])
        average_loss.append(loss.item())
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print("AVG: ", sum(average_loss) / len(average_loss))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


val_loss = validate_model(model, XValid_tensor, YValid_tensor, criterion, num_layers, hidden_size)
print(f'Validation Loss: {val_loss}')

torch.save(model.state_dict(), 'model_state_dict.pth')

#model.load_state_dict(torch.load('model_state_dict.pth'))
#model.eval()

hidden = (torch.zeros(num_layers, 1, hidden_size),
           torch.zeros(num_layers, 1, hidden_size))

predictions = []
for i in tqdm(range(len(X_Train))):
     out, hidden = model(XTrain_tensor[i].unsqueeze(1), hidden)
     predictions.append(out.item())

plt.plot(Y_Train, label='Actual', linewidth=0.75)
plt.plot(predictions, label='Predicted', linewidth=0.75)
plt.ylabel('Price in USD')
plt.xlabel('Samples')
plt.legend()
plt.show()

