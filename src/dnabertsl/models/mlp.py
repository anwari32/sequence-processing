import torch

class MLP(torch.nn.Module):
    """
    Baseline architecture for sequential labelling.
    """
    def __init__(self, bert=None, config=None):
        super().__init__()
        self.num_labels = config.get("num_labels", 8) if config else 8
        self.num_layers = config.get("num_layers", 1) if config else 1
        self.hidden_size = config.get("hidden_size", 768) if config else 768
        self.input_layer = torch.nn.Linear(1, self.hidden_size)
        self.hidden_layer = torch.nn.Sequential()
        for i in range(self.num_layers):
            self.hidden_layer.add_module(
                f"linear-{i}", torch.nn.Linear(self.hidden_size, self.hidden_size)
            )
            self.hidden_layer.add_module(
                f"relu-{i}", torch.nn.ReLU()
            )
        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.activation = torch.nn.Softmax(dim=2)

    def forward(self, input):
        output = self.input_layer(input)
        output = self.hidden_layer(output)
        output = self.classifier(output)
        output = self.activation(output)
        return output


if __name__ == "__main__":
    sequence_length = 512
    sequence_dim = 1
    input_ids = torch.randint(0, 69, (5, sequence_length, sequence_dim)).float().to("cuda:0")
    target_label = torch.randint(0, 8, (5, sequence_length)).to("cuda:0")
    criterion = torch.nn.CrossEntropyLoss()

    model = Baseline()
    model = model.to("cuda:0")
    output = model(input_ids)
    loss = criterion(output.view(-1, 8), target_label.view(-1))
    print(output, output.shape)
    print(loss)
