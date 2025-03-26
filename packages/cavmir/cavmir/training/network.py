import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CAVNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape: int,
        target_shape: int,
        dropout_rate: float = 0.0,
    ):
        """
        Parameters:
        ----------
        input_shape: int
            The number of input features.
        target_shape: int
            The number of output features (target dimensions).
        dropout_rate: float
            The dropout rate to apply during training.
        """
        super(CAVNetwork, self).__init__()

        self.linear = torch.nn.Linear(input_shape, target_shape)
        self.dropout = torch.nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor):
        """
        Forward pass for inference.
        """
        logits = self.linear(x)
        return torch.sigmoid(logits)

    def forward_train(self, x: torch.Tensor):
        """
        Forward pass for training (dropout will be active).
        """
        if self.dropout:
            x = self.dropout(x)
        logits = self.linear(x)
        return logits

    def get_concept_activation_vector(self) -> np.ndarray:
        # Get the weights of the first layer
        return self.linear.cpu().weight.detach().numpy()

    def set_concept_activation_vector(
        self, concept_activation_vector: np.ndarray
    ) -> None:
        # Set the weights of the first layer
        self.linear.weight = torch.nn.Parameter(
            torch.tensor(concept_activation_vector).float()
        )
        self.linear.to(DEVICE)
