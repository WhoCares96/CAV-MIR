import numpy as np
import torch


class SimpleDenseNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape: int,
        target_shape: int,
        hidden_layers: list[int] = [],
        dropout_rate: float = 0.0,
        nonlinear_last_layer: bool = False,
    ):
        """
        Parameters:
        ----------
        input_shape: int
            The number of input features.
        target_shape: int
            The number of output features (target dimensions).
        hidden_layers: list[int]
            A list of integers defining the number of units in the hidden layers. Can be an empty list.
        dropout_rate: float
            The dropout rate to apply after each layer. Applies at least once if hidden_layers is empty.
        nonlinear_last_layer: bool
            Whether to use a nonlinear activation function in the last layer. Defaults to False (linear).
        """
        super(SimpleDenseNetwork, self).__init__()

        self.hidden_layers = hidden_layers

        # Define the layer widths by combining input, hidden, and target layers
        layer_widths = [input_shape] + hidden_layers + [target_shape]

        layers = []
        for i in range(len(layer_widths) - 1):
            # Add a fully connected (linear) layer
            layers.append(torch.nn.Linear(layer_widths[i], layer_widths[i + 1]))

            if i < len(layer_widths) - 2:  # Nonlinear activation for all but last layer
                layers.append(torch.nn.ReLU())

                # Add dropout if specified
                if dropout_rate > 0.0:
                    layers.append(torch.nn.Dropout(dropout_rate))

        # Add nonlinear activation to the last layer if specified
        if nonlinear_last_layer:
            layers.append(torch.nn.ReLU())

        # Ensure at least one dropout layer if no hidden layers exist
        if not hidden_layers and dropout_rate > 0.0:
            layers.append(torch.nn.Dropout(dropout_rate))

        # Create the sequential model
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for inference.
        """
        logits = self.network(x)  # Get raw logits
        return torch.sigmoid(logits)  # Apply sigmoid to get probabilities for inference

    def forward_train(self, x: torch.Tensor):
        """
        Forward pass for training (dropout will be active).
        """
        return self.network(x)  # Return raw logits

    def get_concept_activation_vector(self) -> np.ndarray:
        if self.hidden_layers:
            raise ValueError(
                "This method is only supported for models without hidden layers."
            )

        # Get the weights of the first layer
        return self.network.cpu()[0].weight.detach().numpy()
