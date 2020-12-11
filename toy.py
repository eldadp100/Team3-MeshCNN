import functools

from models.layers import mesh
import argparse

from models.layers.mesh_circular_layer import CircularMeshLSTM
from models.layers.mesh_pool import MeshPool
from models.networks import MeshConvNet, MResConv
from models.layers.mesh_pool_sa import MeshPoolSA


parser = argparse.ArgumentParser()
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--save_latest_freq', type=int, default=250, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1,
                    help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--run_test_freq', type=int, default=1, help='frequency of running test in training script')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1,
                    help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--which_epoch', type=str, default='latest',
                    help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=500,
                    help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--num_aug', type=int, default=10, help='# of augmentation files')
parser.add_argument('--scale_verts', action='store_true', help='non-uniformly scale the mesh e.g., in x, y or z')
parser.add_argument('--slide_verts', type=float, default=0,
                    help='percent vertices which will be shifted along the mesh surface')
parser.add_argument('--flip_edges', type=float, default=0, help='percent of edges to randomly flip')
parser.add_argument('--no_vis', action='store_true', help='will not use tensorboard')
parser.add_argument('--verbose_plot', action='store_true', help='plots network weights, etc.')
parser.add_argument('--dataroot')
parser.add_argument('--name')
parser.add_argument('--pool_res')
parser.add_argument('--norm')
parser.add_argument('--resblocks')

opt = parser.parse_args()

mesh_path = 'tree_98.obj'
mesh = mesh.Mesh(file=mesh_path, opt=opt, export_folder='.')
print(mesh.edges_count)

import torch
from models.layers.mesh_conv import MeshConv, MeshEdgeEmbeddingLayer
from models.layers.mesh_self_attention import MeshSelfAttention

# from models.layers.mesh_conv_old import MeshConv
sa = MeshSelfAttention(5, 5, 10)
mc = MeshConv(5, 15)
# a = torch.rand(1, 750, 5)
import pickle

with open('input_a.p', 'rb') as f:
    a = torch.load(f, map_location='cpu')
embd_layer = MeshEdgeEmbeddingLayer(5, 10)
print(a['x'].shape)
print(embd_layer(a['x']).shape)
print(sa(a['x'])[0].shape)


print(a['x'].shape)
o = mc(a['x'], a['mesh'])
print(o.shape)

from torch import nn

norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=1)


class MeshTransformerNet(nn.Module):
    """ Mesh Transformer """

    def __init__(self, embd_size=16, sa_window_size=10):  # TODO: sa_window_size
        super(MeshTransformerNet, self).__init__()
        self.k = [5, 10, 20, 40]
        self.res = [600, 450, 300, 210]
        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'sa{}'.format(i), MeshSelfAttention(ki, 64, 10))
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1]))
            setattr(self, 'sa_pool{}'.format(i), MeshPoolSA(self.res[i + 1]))
        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc = nn.Linear(self.k[-1], 35)
        self.relu = nn.ReLU()

    def forward(self, x, mesh):
        for i in range(len(self.k) - 1):
            x, sa_mat = getattr(self, 'sa{}'.format(i))(x)
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            # x = nn.ReLU()(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'sa_pool{}'.format(i))(x, mesh, sa_mat)
            # x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = self.fc(x)
        return x


net = MeshTransformerNet()
print(net(a['x'], a['mesh']).shape)



