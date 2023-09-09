import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLinearNN(nn.Module):
    def __init__(self):
        super(SimpleLinearNN, self).__init__()

        # Flatten the input and feed it into a hidden layer
        self.fc1 = nn.Linear(200 * 12, 500)  # Chose 500 hidden units arbitrarily

        # Hidden layer to Output layer
        # Chose an output size of 100 arbitrarily, adjust according to your problem
        self.fc2 = nn.Linear(500, 100) 

    def forward(self, x):
        # Flatten the input data
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x