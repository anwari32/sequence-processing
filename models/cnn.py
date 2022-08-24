import torch
import json

class ConvBlock(torch.nn.Module):
    def __init__(self, n_filters, kernel_size):
        super().__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size

    def forward(self, input_ids):
        return None

class CNN(torch.nn.Module):

    def __init__(self, config=None):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(4, 16, 7)
        self.conv2 = torch.nn.Conv1d(16, 32, 6)
        self.conv3 = torch.nn.Conv1d(32, 64, 6)
        self.relu = torch.nn.ReLU()
        self.max_pooling = torch.nn.MaxPool1d(2, stride=2)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.flatten = torch.nn.Flatten()
        self.hidden_layer = torch.nn.Linear(64, 100)
        self.classification_layer = torch.nn.Linear(100, 8)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, input_ids):
        output = self.conv1(input_ids)
        output = self.relu(output)
        output = self.max_pooling(output)
        output = self.dropout(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.max_pooling(output)
        output = self.dropout(output)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.max_pooling(output)
        output = self.dropout(output)
        output = self.flatten(output)
        # output = self.hidden_layer(output)
        # output = self.relu(output)
        # output = self.classification_layer(output)
        # output = self.activation(output)

        return output

if __name__ == "__main__":
    m = CNN()
    batch_size = 5
    sequence_length = 512
    channel_size = 4
    input_ids = torch.randn(channel_size, sequence_length)
    output = m(input_ids)
    print(output, output.shape)