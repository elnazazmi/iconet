import torch
import torch.nn as nn

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


class MultiFeatureLSTM(nn.Module):
    def __init__(self, n_features, n_hidden=15, n_layers=1):
        super(MultiFeatureLSTM, self).__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
            dropout=0
        )

        self.regressor = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]

        return self.regressor(out)
