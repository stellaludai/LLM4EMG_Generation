import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim = 12, num_channels = 32, output_dim = None):
        super(CNN, self).__init__()
        
        # Define the 1D convolutional layers
        self.conv1 = nn.Conv1d(input_dim, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels*2, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layers
        # Assuming input sequence length is 200, and after two max pooling operations it becomes 50
        self.fc1 = nn.Linear(num_channels*2*50, 100) 
        self.fc2 = nn.Linear(100, output_dim)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input shape: (batch_size, seq_length, input_dim)
        # CNN expects: (batch_size, input_dim, seq_length)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


