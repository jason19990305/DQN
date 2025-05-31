import torch.nn as nn

class Network(nn.Module):
    def __init__(self, args, hidden_layers=[64, 64]):
        super(Network, self).__init__()

        self.num_states = args.num_states
        self.num_actions = args.num_actions

        # Insert input and output sizes into hidden_layers
        hidden_layers.insert(0, self.num_states)
        hidden_layers.append(self.num_actions)

        # Create fully connected layers
        fc_list = []
        for i in range(len(hidden_layers) - 1):
            num_input = hidden_layers[i]
            num_output = hidden_layers[i + 1]
            layer = nn.Linear(num_input, num_output)
            fc_list.append(layer)

        # Convert list to ModuleList for proper registration
        self.layers = nn.ModuleList(fc_list)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through all layers except the last, applying ReLU activation
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
        # The final layer outputs the Q-value directly (no activation)
        q_value = self.layers[-1](x)
        return q_value
