import functools

from models.layers import mesh
import argparse

from models.layers.mesh_circular_layer import CircularMeshLSTM
from models.layers.mesh_pool import MeshPool
from models.networks import MeshConvNet

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
from models.layers.mesh_self_attention import MeshSA

# from models.layers.mesh_conv_old import MeshConv
sa = MeshSA(5, 5, 10)
mc = MeshConv(5, 15)
# a = torch.rand(1, 750, 5)
import pickle

with open('input_a.p', 'rb') as f:
    a = torch.load(f, map_location='cpu')
embd_layer = MeshEdgeEmbeddingLayer(5, 10)
print(a['x'].shape)
print(embd_layer(a['x']).shape)
print(sa(a['x']).shape)


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

        self.edges_embedding = MeshEdgeEmbeddingLayer(self.k[0], embd_size)
        self.k[0] = embd_size
        self.sa_layer = MeshSA(self.k[0], self.k[0], window_size=sa_window_size)
        self.dropout = nn.Dropout(p=0.2)
        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'sa{}'.format(i), MeshSA(ki, ki, window_size=sa_window_size))
            setattr(self, 'cirLSTM{}'.format(i), CircularMeshLSTM(ki, 220, self.k[i+1], 1))
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
            x = getattr(self, 'cirLSTM{}'.format(i))(x)
            # x = getattr(self, 'conv{}'.format(i))(x, mesh)
            # x = self.relu(x)
            # x = self.dropout(x)
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

net = MeshTransformerNet()
print(net(a['x'], a['mesh']).shape)




#
#
#
#
#
#
#
#
# class MeshEncoder(nn.Module):
#     def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
#         super(MeshEncoder, self).__init__()
#         self.fcs = None
#         self.convs = []
#         for i in range(len(convs) - 1):
#             if i + 1 < len(pools):
#                 pool = pools[i + 1]
#             else:
#                 pool = 0
#             self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
#         self.global_pool = None
#         if fcs is not None:
#             self.fcs = []
#             self.fcs_bn = []
#             last_length = convs[-1]
#             if global_pool is not None:
#                 if global_pool == 'max':
#                     self.global_pool = nn.MaxPool1d(pools[-1])
#                 elif global_pool == 'avg':
#                     self.global_pool = nn.AvgPool1d(pools[-1])
#                 else:
#                     assert False, 'global_pool %s is not defined' % global_pool
#             else:
#                 last_length *= pools[-1]
#             if fcs[0] == last_length:
#                 fcs = fcs[1:]
#             for length in fcs:
#                 self.fcs.append(nn.Linear(last_length, length))
#                 self.fcs_bn.append(nn.InstanceNorm1d(length))
#                 last_length = length
#             self.fcs = nn.ModuleList(self.fcs)
#             self.fcs_bn = nn.ModuleList(self.fcs_bn)
#         self.convs = nn.ModuleList(self.convs)
#         reset_params(self)
#
#     def forward(self, x):
#         fe, meshes = x
#         encoder_outs = []
#         for conv in self.convs:
#             fe, before_pool = conv((fe, meshes))
#             encoder_outs.append(before_pool)
#         if self.fcs is not None:
#             if self.global_pool is not None:
#                 fe = self.global_pool(fe)
#             fe = fe.contiguous().view(fe.size()[0], -1)
#             for i in range(len(self.fcs)):
#                 fe = self.fcs[i](fe)
#                 if self.fcs_bn:
#                     x = fe.unsqueeze(1)
#                     fe = self.fcs_bn[i](x).squeeze(1)
#                 if i < len(self.fcs) - 1:
#                     fe = F.relu(fe)
#         return fe, encoder_outs
#
#     def __call__(self, x):
#         return self.forward(x)
#
#
# class MeshDecoder(nn.Module):
#     def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
#         super(MeshDecoder, self).__init__()
#         self.up_convs = []
#         for i in range(len(convs) - 2):
#             if i < len(unrolls):
#                 unroll = unrolls[i]
#             else:
#                 unroll = 0
#             self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
#                                         batch_norm=batch_norm, transfer_data=transfer_data))
#         self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
#                                  batch_norm=batch_norm, transfer_data=False)
#         self.up_convs = nn.ModuleList(self.up_convs)
#         reset_params(self)
#
#     def forward(self, x, encoder_outs=None):
#         fe, meshes = x
#         for i, up_conv in enumerate(self.up_convs):
#             before_pool = None
#             if encoder_outs is not None:
#                 before_pool = encoder_outs[-(i+2)]
#             fe = up_conv((fe, meshes), before_pool)
#         fe = self.final_conv((fe, meshes))
#         return fe
#
#     def __call__(self, x, encoder_outs=None):
#         return self.forward(x, encoder_outs)
