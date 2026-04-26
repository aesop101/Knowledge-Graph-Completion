import torch
import torch.nn as nn
import math

class ISE_TE(nn.Module):
    def __init__(self, input_size, num_head, hidden_size, num_layers):
        super(ISE_TE, self).__init__()
        
        self.d_model = 4 * input_size
        self.input_projection = nn.Linear(input_size, self.d_model)
        
        blending_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_head,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.blender = nn.TransformerEncoder(blending_layer, num_layers=1)
        self.linear2 = nn.Linear(input_size, input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_head,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.TE = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.activation = nn.Sigmoid()

        self.linear3 = nn.Linear(self.d_model, self.d_model)
        self.linear4 = nn.Linear(self.d_model, self.d_model)
        self.linear5 = nn.Linear(2 * self.d_model, input_size)

    def blend_func(self, e, r):
        e_proj = self.input_projection(e).unsqueeze(1) 
        r_proj = self.input_projection(r).unsqueeze(1)

        seq = torch.cat((e_proj, r_proj), dim=1)
        blended_seq = self.blender(seq)
        return torch.mean(blended_seq, dim=1)

    def forward(self, neighbors_h, neighbors_t, e_h, e_t, mask_h=None, mask_t=None):
        r_h = self.linear2(e_t - e_h)
        r_t = self.linear2(e_h - e_t)

        B = e_h.size(0)
        D_in = e_h.size(-1)

        he_h = self.activation(self.blend_func(e_h, r_h)).unsqueeze(1)
        he_t = self.activation(self.blend_func(e_t, r_t)).unsqueeze(1)

        Nh = neighbors_h.size(1)
        nei_h_e, nei_h_r = torch.split(neighbors_h, D_in, dim=-1)
        hni_h_flat = self.activation(self.blend_func(nei_h_e.reshape(-1, D_in), 
                                                     nei_h_r.reshape(-1, D_in)))
        hni_h = hni_h_flat.view(B, Nh, -1) 
        Nt = neighbors_t.size(1)
        nei_t_e, nei_t_r = torch.split(neighbors_t, D_in, dim=-1)
        hni_t_flat = self.activation(self.blend_func(nei_t_e.reshape(-1, D_in), 
                                                     nei_t_r.reshape(-1, D_in)))
        hni_t = hni_t_flat.view(B, Nt, -1) 
        z0_h = torch.cat((he_h, hni_h), dim=1)
        z0_t = torch.cat((he_t, hni_t), dim=1)

        zL_h = self.TE(z0_h, src_key_padding_mask=mask_h)
        zL_t = self.TE(z0_t, src_key_padding_mask=mask_t)

        def aggregate(zL, mask):
            if mask is not None:
                zL_masked = zL.masked_fill(mask.unsqueeze(-1), 0.0)
                valid_len = (~mask).sum(dim=1, keepdim=True).float()
                ze = zL_masked.sum(dim=1) / (valid_len + 1e-8)
            else:
                ze = torch.mean(zL, dim=1)
            zL_0 = zL[:, 0, :] 
            return self.activation(self.linear4(self.activation(self.linear3(zL_0)) + ze))

        oh = aggregate(zL_h, mask_h)
        ot = aggregate(zL_t, mask_t)

        return self.activation(self.linear5(torch.cat((oh, ot), dim=-1)))