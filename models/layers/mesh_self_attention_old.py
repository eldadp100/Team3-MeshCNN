import torch
import torch.nn as nn
import math


class SA_Layer(nn.Module):
    def __init__(self, elem_size, embd_size, use_V=False, use_res_conn=True):
        super(SA_Layer, self).__init__()
        self.use_V = use_V
        self.use_res_conn = use_res_conn if not use_V else False
        # self.res_dropout = nn.Dropout(p=0.2)
        self.res_dropout = None

        self.embd_size = embd_size
        self.query_lin = nn.Linear(elem_size, embd_size)
        self.key_lin = nn.Linear(elem_size, embd_size)
        if self.use_V:
            self.value_lin = nn.Linear(elem_size, embd_size)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        seq_queries = self.relu(self.query_lin(x))
        seq_keys = self.relu(self.key_lin(x))
        output_element_size = self.embd_size if self.use_V else elem_size

        attention_mat = torch.matmul(seq_queries, seq_keys.permute(0, 2, 1)) / math.sqrt(elem_size)
        attention = self.softmax(attention_mat)  # softmax for each row
        new_values = torch.matmul(attention, x)
        out = new_values

        # out = torch.empty(N, seq_size, output_element_size).to(x.device)  # empty to save grads
        # for i in range(N):
        #     curr_q = seq_queries[i]
        #     curr_k = seq_keys[i]
        #     if self.use_V:
        #         curr_v = self.relu(self.value_lin(x[i]))
        #     else:
        #         curr_v = x[i]
        #     attention_mat = torch.mm(curr_q, torch.transpose(curr_k, 1, 0)) / math.sqrt(elem_size)
        #     attention = self.softmax(attention_mat)  # softmax for each row
        #     new_values = torch.mm(attention, curr_v)
        #     out[i] = new_values  # + self.res_con_p * curr_v

        if self.use_res_conn:
            if self.res_dropout is not None:
                x = self.res_dropout(x)
            out = x + out
        return out


class Patched_SA_Layer(nn.Module):
    def __init__(self, elem_size, embd_size, window_size, use_V=False):
        super(Patched_SA_Layer, self).__init__()
        self.sa_layer = SA_Layer(elem_size, embd_size, use_V=use_V)
        self.window_size = window_size
        self.embd_size = embd_size
        self.use_V = use_V

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        output_element_size = self.embd_size if self.use_V else elem_size
        out = torch.empty(N, seq_size, output_element_size)
        patches_num = seq_size // self.window_size
        add_patch = False
        if seq_size % self.window_size != 0:
            add_patch = True

        for i in range(N):
            curr_x = x[i]
            curr_x_patches = curr_x[:patches_num * self.window_size].reshape(patches_num, self.window_size, elem_size)
            if add_patch:
                rest_seq = curr_x[patches_num * self.window_size:]
                rest_seq_padded = torch.zeros(1, curr_x_patches.shape[1], curr_x_patches.shape[2]).to(x.device)
                rest_seq_padded[0, :len(rest_seq), :] = rest_seq
                curr_x_patches = torch.cat([curr_x_patches, rest_seq_padded], dim=0)
            # curr_x_patches.requires_grad = x.requires_grad
            curr_x_patches = self.sa_layer(curr_x_patches)  # get patches as batch

            out[i, :] = curr_x_patches.reshape(curr_x_patches.shape[0] * curr_x_patches.shape[1],
                                               curr_x_patches.shape[2])[:seq_size]

        return out


class MeshSA(nn.Module):
    def __init__(self, in_size, embd_size, window_size, heads=2):
        super(MeshSA, self).__init__()
        self.heads = heads
        self.sa_heads = nn.ModuleList()
        for _ in range(heads):
            if window_size is None:
                sa_layer = SA_Layer(in_size, embd_size)
            else:
                sa_layer = Patched_SA_Layer(in_size, embd_size, window_size)
            self.sa_heads.append(sa_layer)
        self.out_lin = nn.Linear(self.heads, 1)  # TODO: init to 1 / heads

    def forward(self, edges):
        batch_size, edges_num, edges_features_num = edges.shape
        device = edges.device
        edges = edges.permute(0, 2, 1)  # put seq in place (before elem)
        out = torch.empty(batch_size, edges_features_num, edges_num, self.heads)
        for i in range(self.heads):
            out[:, :, :, i] = self.sa_heads[i](edges)
        out = self.out_lin(out).squeeze(-1)
        return out.permute(0, 2, 1).to(device)  # permute back


if __name__ == '__main__':
    sa_layer = MeshSA(10, 21, 12)
    a = torch.rand(32, 10, 100)  # batch_size x elem_size x seq len. seq of 100 elements
    print(sa_layer(a).shape)