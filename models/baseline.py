import torch

class Baseline(torch.nn.Module):
    """
    Baseline architecture for sequential labelling.
    """
    def __init__(self, config=None):
        super().__init__()
        self.num_labels = config.get("num_labels", 8) if config else 8
        self.num_layers = config.get("num_layers", 1) if config else 1
        self.hidden_size = config.get("hidden_size", 512) if config else 512
        self.input_layer = torch.nn.Linear(1, self.hidden_size)
        self.hidden_layer = torch.nn.Sequential()
        for i in range(self.num_layers):
            self.hidden_layer.add_module(
                f"linear-{i}", torch.nn.Linear(self.hidden_size, self.hidden_size)
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
    input_sample = torch.rand(5, 512, 1)
    model = Baseline()
    output = model(input_sample)
    print(output.shape)
    print(output[0][0])
