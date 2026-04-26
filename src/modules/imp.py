import torch
import torch.nn as nn
import math

def calculate_score(p_q, p_s):
    D = p_q.size(-1)

    scores = torch.bmm(p_q, p_s.transpose(1, 2)).squeeze(1)
    attn_weights = torch.softmax(scores / math.sqrt(D), dim=-1)

    prototype = torch.bmm(attn_weights.unsqueeze(1), p_s).squeeze(1)
    final_score = (p_q.squeeze(1) * prototype).sum(dim=-1)
    return final_score


class IMP(nn.Module):
    def __init__(self, input_size, num_head, hidden_size, num_layers):
        super(IMP, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_head,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.TE = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, p_q, p_s, mask=None):
        c0 = torch.cat([p_q, p_s], dim=1) # [B, 1+K, D]
        c = self.TE(c0, src_key_padding_mask=mask)
  
        D = p_q.size(-1)
        query_orig = p_q.squeeze(1) 
        support_inter = c[:, 1:, :]  

        delta = torch.bmm(c[:, 0, :].unsqueeze(1), support_inter.transpose(1, 2)).squeeze(1)
        attn_weights = torch.softmax(delta / math.sqrt(D), dim=-1)

        rho_r = torch.bmm(attn_weights.unsqueeze(1), p_s).squeeze(1)
        final_score = torch.sum(c[:, 0, :].squeeze(1) * rho_r, dim=-1)
        return final_score