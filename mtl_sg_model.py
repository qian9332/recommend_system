import torch.nn as nn

class SceneEncoder(nn.Module):
    def __init__(self, resource_emb_dim, realtime_dim, scene_dim):
        super(SceneEncoder, self).__init__()
        # Update MLP input dimension
        self.mlp = nn.Sequential(
            nn.Linear(resource_emb_dim + realtime_dim + scene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the encoder.
        Args:
            x (Tensor): Input tensor
        Returns:
            Tensor: Encoded output
        """
        return self.mlp(x)