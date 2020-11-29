import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshLinearLayer(nn.Module):

    def __init__(self, input_size, output_size, bias=True):
        super(MeshLinearLayer, self).__init__()
        self.lin = nn.Linear(input_size, output_size, bias=bias)
        self.out_size = output_size

    def forward(self, x):
        # x = x.squeeze(-1)
        out = torch.empty(x.shape[0], x.shape[1], self.out_size)
        for i in range(x.shape[0]):
            out[i] = self.lin(x[i])
        return out


class MeshEdgeEmbeddingLayer(nn.Module):
    """
    Very important - who said that a-c is meaningfull at first layer...
    """

    def __init__(self, input_size, embedding_size, bias=True):
        super(MeshEdgeEmbeddingLayer, self).__init__()
        self.lin = nn.Linear(input_size, embedding_size, bias=bias)

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        return self.lin(x).permute(0, 2, 1)


class MeshConv(nn.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    x: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """

    def __init__(self, edge_in_feat, edge_out_feat, kernel_size=1, bias=True):
        super(MeshConv, self).__init__()
        # we support only kernel_size=1...
        self.lin = MeshLinearLayer(edge_in_feat * 5, edge_out_feat, bias=bias)

    def __call__(self, edge_f, mesh):
        return self.forward(edge_f, mesh)

    def forward(self, x, mesh):
        device = x.device
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        G = G.permute(0, 2, 1, 3)
        x = G.reshape(G.shape[0], G.shape[1], -1)
        x = self.lin(x)
        return x.permute(0, 2, 1).unsqueeze(-1).to(device)

    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1  # shift

        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts one-ring neighbors (4x) -> m.gemm_edges
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., xsz x 5
        """
        padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm

#
#
# class MeshConv(nn.Module):
#     """
#         Computes convolution between edges and 4 incident (1-ring) edge neighbors
#         in the forward pass takes:
#         x: edge features (Batch x Features x Edges)
#         mesh: list of mesh data-structure (len(mesh) == Batch)
#         and applies convolution
#     """
#
#     # TODO: separate SA layer and Conv layer - copy the methods of conv to SA.
#     def __init__(self, in_size, out_size, bias=True, kernel_size=1):
#         super(MeshConv, self).__init__()
#         self.kernel_size = kernel_size  # which ring
#         self.fc = nn.Linear(in_size, out_size, bias=bias)
#
#     def __call__(self, edge_f, mesh):
#         return self.forward(edge_f, mesh)
#
#     def forward(self, x, mesh):
#         # attention phase (split later - easy)
#         # treat each mesh by it self for now. The sequnce is the edges!!! (already sorted)
#         # take window of 40 (better to depend on kernel_size but it isn't a must) and compute it...
#
#         print(x.shape)
#         batch_size = x.shape[0]
#         num_edges = x.shape[1]
#         num_features = x.shape[2]
#         x = x.squeeze(-1)
#         G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
#         # build 'neighborhood' and apply convolution - similar to GCN conv
#         f = self.create_GeMM(x, G)
#         x_1 = f[:, :, :, 1] + f[:, :, :, 3]
#         x_2 = f[:, :, :, 2] + f[:, :, :, 4]
#         x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
#         x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
#         f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
#         x = G.reshape(x.shape[0], -1)
#
#         print(x.shape)
#         x = self.fc(x)
#         x = x.reshape(x.shape[0], 1, -1)  # TODO: replace with unsqueeze(1)
#         print(x)
#         return x
#
#     def flatten_gemm_inds(self, Gi):
#         (b, ne, nn) = Gi.shape
#         ne += 1
#         batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
#         add_fac = batch_n * ne
#         add_fac = add_fac.view(b, ne, 1)
#         add_fac = add_fac.repeat(1, 1, nn)
#         # flatten Gi
#         Gi = Gi.float() + add_fac[:, 1:, :]
#         return Gi
#
#     def create_GeMM(self, x, Gi):
#         """ gathers the edge features (x) with from the 1-ring indices (Gi)
#         applys symmetric functions to handle order invariance
#         returns a 'fake image' which can use 2d convolution on
#         output dimensions: Batch x Channels x Edges x 5
#         """
#         Gishape = Gi.shape
#         # pad the first row of  every sample in batch with zeros
#         padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
#         # padding = padding.to(x.device)
#         x = torch.cat((padding, x), dim=2)
#         Gi = Gi + 1  # shift
#
#         # first flatten indices
#         Gi_flat = self.flatten_gemm_inds(Gi)
#         Gi_flat = Gi_flat.view(-1).long()
#         #
#         odim = x.shape
#         x = x.permute(0, 2, 1).contiguous()
#         x = x.view(odim[0] * odim[2], odim[1])
#
#         f = torch.index_select(x, dim=0, index=Gi_flat)
#         f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
#         f = f.permute(0, 3, 1, 2)
#
#         # _f = f[:, :, :, 0]
#         # if self.kernel_size == 1:
#         #     for i in range(1, 5):
#         #         _f += f[:, :, :, i]
#         #     _f /= 5.0
#         # elif self.kernel_size == 2:
#         #     if 8:  # TODO: detect if ring is 8 edges or 12 edges
#         #         for i in range(1, 5):
#         #             _f += f[:, :, :, i]
#         #         _f /= 8.0
#         #     elif 12:
#         #         for i in range(1, 5):
#         #             _f += f[:, :, :, i]
#         #         _f /= 12.0
#         # else:
#         #     raise Exception(f"kernel size={self.kernel_size} not supported. kernel_size should be 1 or 2")
#
#         # apply the symmetric functions for an equivariant conv
#         x_1 = f[:, :, :, 1] + f[:, :, :, 3]
#         x_2 = f[:, :, :, 2] + f[:, :, :, 4]
#         x_3 = torch.abs(f[:, :, :, 1] - f[:, :, :, 3])
#         x_4 = torch.abs(f[:, :, :, 2] - f[:, :, :, 4])
#         f = torch.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], dim=3)
#         # print(_f.shape)
#         # return _f
#         return f
#
#     def pad_gemm(self, m, xsz, device):
#         """ extracts one-ring neighbors (4x) -> m.gemm_edges
#         which is of size #edges x 4
#         add the edge_id itself to make #edges x 5
#         then pad to desired size e.g., xsz x 5
#         """
#         padded_gemm = torch.tensor(m.gemm_edges, device=device).float()
#         padded_gemm = padded_gemm.requires_grad_()
#         padded_gemm = torch.cat((torch.arange(m.edges_count, device=device).float().unsqueeze(1), padded_gemm), dim=1)
#         # pad using F
#         padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.edges_count), "constant", 0)
#         padded_gemm = padded_gemm.unsqueeze(0)
#         return padded_gemm
#
