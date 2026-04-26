import torch
import torch.nn as nn

class ISE(nn.Module):
    def __init__(self, input_size, num_head, hidden_size, num_layers):
        super(ISE, self).__init__()
        self.d_model = 4 * input_size
        self.linear1 = nn.Linear(2 * input_size, self.d_model)

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

    def forward(self, neighbors_h, neighbors_t, e_h, e_t, mask_h=None, mask_t=None):
        r_h = self.linear2(e_t - e_h)
        r_t = self.linear2(e_h - e_t)

        he_h = self.activation(self.linear1(torch.cat((e_h, r_h), dim=-1))).unsqueeze(1)
        he_t = self.activation(self.linear1(torch.cat((e_t, r_t), dim=-1))).unsqueeze(1)

        hni_h = self.activation(self.linear1(neighbors_h))
        hni_t = self.activation(self.linear1(neighbors_t))
 
        z0_h = torch.cat((he_h, hni_h), dim=1)
        z0_t = torch.cat((he_t, hni_t), dim=1)

        zL_h = self.TE(z0_h, src_key_padding_mask=mask_h)
        zL_t = self.TE(z0_t, src_key_padding_mask=mask_t)

        def aggregate(zL, mask):
            if mask is not None:
                zL_masked = zL.masked_fill(mask.unsqueeze(-1), 0.0)
                valid_len = (~mask).sum(dim=1, keepdim=True).float()
                ze = zL_masked.sum(dim=1) / valid_len
            else:
                ze = torch.mean(zL, dim=1)

            zL_0 = zL[:, 0, :]
            term1 = self.activation(self.linear3(zL_0)) + ze
            term2 = self.linear4(term1)
            return self.activation(term2)

        oh = aggregate(zL_h, mask_h)
        ot = aggregate(zL_t, mask_t)

        p = self.activation(self.linear5(torch.cat((oh, ot), dim=-1)))
        return p