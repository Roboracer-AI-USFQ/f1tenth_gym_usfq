import torch
import torch.nn as nn

class Encoder(torch.nn.Module):
    """
    this is the encoding module
    """

    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=1,
        concat_action=False,
        dropout=0.0,
    ):
        """
        state_dim: the state dimension
        stacked_frames: #timesteps considered in history
        hidden_size: hidden layer size
        num_layers: how many layers

        the input state should be of size [batch, stacked_frames, state_dim]
        the output should be of size [batch, hidden_size]
        """
        super().__init__()
        self.hidden_size = self.feature_dim = hidden_size

    def get_feature_dim(self):
        return self.feature_dim
    
    def forward(self, states, actions=None):
        return None

class CNNEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=1,
        concat_action=False,
        dropout=0.0,):
        super().__init__(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            history_length=history_length,
            concat_action=concat_action,
            dropout=dropout,
        )
        print(self.feature_dim)
        #print(input_dim)
        layers = []
        if num_layers > 1:
            for i in range(num_layers - 1):
                input_channel = hidden_size if i > 0 else input_dim[0]
                layers.append(
                    nn.Conv1d(
                        in_channels=input_channel,
                        out_channels=hidden_size,
                        kernel_size=3,
                        padding=1,
                    )
                )
                layers.append(nn.ReLU())

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=history_length,
                        padding=0,
                    ),
                    nn.ReLU(),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=hidden_size,
                        kernel_size=history_length,
                        padding=0,
                    ),
                    nn.ReLU(),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, state_dim, seq_len]
        x = self.net(x)
        return x.squeeze(-1)