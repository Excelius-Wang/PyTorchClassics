import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # 输入维度
        self.embed_size = embed_size

        # Q K V 线性变换矩阵
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        # 另外的注意力 Dropout
        self.attn_dropout = nn.Dropout(0.1)
        # 输出线性变换
        self.output_linear = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, embed_size]
        # 线性变换生成 Q K V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 1. 计算注意力分数
        # Q: [batch_size, seq_len, embed_size]
        # K.transpose(1, 2): [batch_size, embed_size, seq_len]
        # attn_scores: [batch_size, seq_len, seq_len]
        attn_scores = torch.matmul(Q, K.transpose(1, 2))
        attn_scores = attn_scores * (self.embed_size ** -0.5)

        # 2. 应用掩码，如果有
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 归一化, softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 4. 应用 Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 计算输出
        # attn_weights: [batch_size, seq_len, seq_len]
        # V: [batch_size, seq_len, embed_size]
        # output: [batch, seq_len, embed_size]
        output = torch.matmul(attn_weights, V)
        # 输出线性变换
        output = self.output_linear(output)

        return output

