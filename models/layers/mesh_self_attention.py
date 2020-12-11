import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['MeshSelfAttention']


class SelfAttentionLayer(nn.Module):
    def __init__(self, elem_size, embd_size):
        super(SelfAttentionLayer, self).__init__()
        self.embd_size = embd_size
        self.query_lin = nn.Linear(elem_size, embd_size)
        self.key_lin = nn.Linear(elem_size, embd_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        N, num_patches, seq_size, elem_size = x.shape
        Q = F.relu(self.query_lin(x))
        K = F.relu(self.key_lin(x))
        attention_mat = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(elem_size)
        attention_mat = F.softmax(attention_mat, dim=-1)  # softmax for each row
        new_values = torch.matmul(attention_mat, x)
        out = new_values
        out = x + out
        return out, attention_mat


class PatchedSelfAttentionLayer(nn.Module):
    def __init__(self, elem_size, embd_size, window_size, use_V=False):
        super(PatchedSelfAttentionLayer, self).__init__()
        self.sa_layer = SelfAttentionLayer(elem_size, embd_size)
        self.window_size = window_size
        self.embd_size = embd_size

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        patches_num = seq_size // self.window_size
        add_patch = False
        if seq_size % self.window_size != 0:
            add_patch = True

        x_patches = x[:, :patches_num * self.window_size, :].reshape(N, patches_num, self.window_size, elem_size)
        if add_patch:
            rest_seq_padding = torch.zeros(N, 1, x_patches.shape[2], x_patches.shape[3]).to(x.device)
            rest_seq_values = x[:, patches_num * self.window_size:, :]
            rest_seq_padding[:, 0, :rest_seq_values.shape[1], :] = rest_seq_values
            x_patches = torch.cat([x_patches, rest_seq_padding], dim=1)
        x_patches, attention_mat = self.sa_layer(x_patches)
        out = x_patches.reshape(x_patches.shape[0], x_patches.shape[1] * x_patches.shape[2],
                                x_patches.shape[3])[:seq_size]
        attention_mat = attention_mat.reshape(attention_mat.shape[0], attention_mat.shape[1] * attention_mat.shape[2],
                                               attention_mat.shape[3])[:seq_size]
        return out, attention_mat



class MeshSelfAttention(nn.Module):
    """
        Multi head memory efficient attention for mesh
    """

    def __init__(self, in_size, embd_size, window_size, heads=2):
        super(MeshSelfAttention, self).__init__()
        self.heads = heads
        self.window_size = window_size
        self.sa_heads = nn.ModuleList()
        for _ in range(heads):
            if window_size is None:
                sa_layer = SelfAttentionLayer(in_size, embd_size)
            else:
                sa_layer = PatchedSelfAttentionLayer(in_size, embd_size, window_size)
            self.sa_heads.append(sa_layer)
        self.out_lin = nn.Linear(self.heads, 1)  # TODO: init to 1 / heads

    def forward(self, edges):
        batch_size, edges_num, edges_features_num = edges.shape
        device = edges.device
        edges = edges.permute(0, 2, 1)  # put seq in place (before elem)
        out = torch.empty((batch_size, edges_features_num, edges_num, self.heads)).to(device)
        for i in range(self.heads):
            attention_mat = torch.zeros((batch_size, edges_features_num, self.window_size)).to(device)
            out[:, :, :, i], att_mat_i = self.sa_heads[i](edges)
            attention_mat += att_mat_i
        attention_mat /= float(self.heads)
        out = self.out_lin(out).squeeze(-1)
        out = F.relu(out)
        return out.permute(0, 2, 1).to(device), attention_mat.to(device)  # permute back


if __name__ == '__main__':
    sa_layer = MeshSelfAttention(10, 21, 12)
    a = torch.rand(32, 10, 100)  # batch_size x elem_size x seq len. seq of 100 elements
    print(sa_layer(a).shape)
