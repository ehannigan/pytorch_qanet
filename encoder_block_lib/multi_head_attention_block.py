from encoder_block_lib.multi_head_attention import MultiHeadAttention
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, num_heads, out_channels, norm):
        super(MultiHeadAttentionBlock, self).__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'layer':
            raise ValueError('no layer normalization implemented yet')
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, d_model=out_channels)

    def forward(self, x, mask=None):
        residual = x.clone()
        x = self.norm(x)
        x = self.multi_head_attention(Q=x, K=x, V=x, mask=mask).permute(0, 2, 1)  #[B, L, d_model]
        return x + residual