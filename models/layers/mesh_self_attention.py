import torch
import torch.nn as nn


class SA_Layer(nn.Module):
    """ for 1 dim sequences """

    def __init__(self, elem_size, embd_size, use_V=False, use_res_conn=True):
        super(SA_Layer, self).__init__()
        self.use_V = use_V
        self.use_res_conn = use_res_conn if not use_V else False
        self.res_dropout = nn.Dropout(p=0.2)

        self.embd_size = embd_size
        self.query_lin = nn.Linear(elem_size, embd_size)
        self.key_lin = nn.Linear(elem_size, embd_size)
        if self.use_V:
            self.value_lin = nn.Linear(elem_size, embd_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N, seq_size, elem_size = x.shape
        seq_queries = self.query_lin(x)
        seq_keys = self.key_lin(x)
        output_element_size = self.embd_size if self.use_V else elem_size
        out = torch.empty(N, seq_size, output_element_size).to(x.device)  # empty to save grads
        attention_matrices = []
        for i in range(N):
            curr_q = seq_queries[i]
            curr_k = seq_keys[i]
            if self.use_V:
                curr_v = self.value_lin(x[i])
            else:
                curr_v = x[i]
            attention_mat = torch.mm(curr_q, torch.transpose(curr_k, 1, 0))
            attention = self.softmax(attention_mat)  # softmax for each row
            attention_matrices.append(attention)
            final_rep = torch.mm(attention, curr_v)
            out[i] = final_rep  # + self.res_con_p * curr_v

        final_att_mat = torch.stack(attention_matrices)

        if self.use_res_conn:
            out = self.res_dropout(x) + out
        return out, final_att_mat


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

        attention_matrix = torch.empty((N, seq_size, self.window_size))

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
            curr_x_patches, attention_matrix_i = self.sa_layer(curr_x_patches)  # get patches as batch

            out[i, :] = curr_x_patches.reshape(curr_x_patches.shape[0] * curr_x_patches.shape[1],
                                               curr_x_patches.shape[2])[:seq_size]
            attention_matrix[i, :, :] = attention_matrix_i.reshape(attention_matrix_i.shape[0] * attention_matrix_i.shape[1],
                                               attention_matrix_i.shape[2])[:seq_size]

        return out, attention_matrix


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

    def forward(self, edges_features):
        device = edges_features.device
        edges_features = edges_features.permute(0, 2, 1)  # put seq in place (before elem)
        out, attention_matrix = self.sa_heads[0](edges_features)
        if self.heads > 1:
            for i in range(1, self.heads):
                out_i, attention_matrix_i = self.sa_heads[i](edges_features)
                out += out_i
                attention_matrix += attention_matrix_i
            attention_matrix /= self.heads
            out /= self.heads
        # return out.permute(0, 2, 1).to(device)  # permute back
        return out.permute(0, 2, 1).to(device), attention_matrix.to(device)



if __name__ == '__main__':
    sa_layer = MeshSA(10, 21, 12)
    a = torch.rand(32, 10, 100)  # batch_size x elem_size x seq len. seq of 100 elements
    print(sa_layer(a).shape)
