import torch
import torch.nn as nn


class SA_Layer(nn.Module):
    """ for 1 dim sequences """

    def __init__(self, elem_size, embd_size):
        super(SA_Layer, self).__init__()
        self.embd_size = embd_size
        self.query_lin = nn.Linear(elem_size, embd_size)
        self.key_lin = nn.Linear(elem_size, embd_size)
        self.value_lin = nn.Linear(elem_size, embd_size)
        self.softmax = nn.Softmax(dim=1)
        # self.res_con_p = nn.Parameter(torch.rand(1) / 5, requires_grad=False)

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        seq_queries = self.query_lin(x)
        seq_keys = self.key_lin(x)
        out = torch.empty(N, seq_size, self.embd_size)  # empty to save grads
        for i in range(N):
            curr_q = seq_queries[i]
            curr_k = seq_keys[i]
            curr_v = self.value_lin(x[i])
            attention_mat = torch.mm(curr_q, torch.transpose(curr_k, 1, 0))
            attention = self.softmax(attention_mat)  # softmax for each row
            final_rep = torch.mm(attention, curr_v)
            out[i] = final_rep  # + self.res_con_p * curr_v
        return out


class Patched_SA_Layer(nn.Module):
    def __init__(self, elem_size, embd_size, window_size):
        super(Patched_SA_Layer, self).__init__()
        self.sa_layer = SA_Layer(elem_size, embd_size)
        self.window_size = window_size
        self.embd_size = embd_size

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        out = torch.empty(N, seq_size, self.embd_size)
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
    def __init__(self, in_size, embd_size, window_size):
        super(MeshSA, self).__init__()
        if window_size is None:
            self.sa_layer = SA_Layer(in_size, embd_size)
        else:
            self.sa_layer = Patched_SA_Layer(in_size, embd_size, window_size)

    def forward(self, edges_features):
        device = edges_features.device
        edges_features = edges_features.permute(0, 2, 1)  # put seq in place (before elem)
        out = self.sa_layer(edges_features)
        return out.permute(0, 2, 1).to(device)  # permute back


if __name__ == '__main__':
    sa_layer = Patched_SA_Layer(10, 7, 21)
    a = torch.rand(32, 100, 10)  # seq of 5 elems
    print(sa_layer(a))
