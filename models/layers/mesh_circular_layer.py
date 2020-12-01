import torch
from torch import nn


class CircularMeshLSTM(nn.Module):
    """
     seq2seq LSTM that changes the edges features by two circular traverses on the mesh edges by fixed order.
     Not bidirectional!!!
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CircularMeshLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.out_lin = nn.Linear(hidden_size, output_size)
        self.lin_hi = nn.Linear(hidden_size, input_size)

    def forward(self, edges):
        edges = edges.permute(0, 2, 1)
        out, tmp = self.lstm(edges)  # first traverse
        out, _ = self.lstm(edges, tmp)  # second traverse
        out = self.out_lin(out)
        return out.permute(0, 2, 1)


# mini test
if __name__ == '__main__':
    cirMeshLSTM = CircularMeshLSTM(10, 7, 25)
    a = torch.rand(32, 10, 100)
    print(cirMeshLSTM(a).shape)
