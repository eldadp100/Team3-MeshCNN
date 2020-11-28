import torch
from torch import nn
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_conv import MeshConv, MeshSA, MeshEdgeEmbeddingLayer


class MeshTransformerNet(nn.Module):
    """ Mesh Transformer """

    def __init__(self, embd_size=16, sa_window_size=10):
        super(MeshTransformerNet, self).__init__()
        self.k = [5, 10, 20, 40]
        self.res = [600, 450, 300, 210]

        self.edges_embedding = MeshEdgeEmbeddingLayer(self.k[0], embd_size)
        self.k[0] = embd_size
        self.sa_layer = MeshSA(self.k[0], self.k[0], window_size=sa_window_size)
        self.dropout = nn.Dropout(p=0.2)
        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'sa{}'.format(i), MeshSA(ki, ki, window_size=sa_window_size))
            setattr(self, 'conv{}'.format(i), MeshConv(ki, self.k[i + 1]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc = nn.Linear(self.k[-1], 35)
        self.relu = nn.ReLU()

    def forward(self, x, mesh):
        x = self.edges_embedding(x)
        x = self.sa_layer(x)
        for i in range(len(self.k) - 1):
            x = getattr(self, 'sa{}'.format(i))(x)
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = self.relu(x)
            x = self.dropout(x)
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])
        x = self.fc(x)
        return x
