import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        # 单头维度
        self.head_dim = embed_size // num_heads

        assert embed_size % num_heads == 0, "Embed_size must be divisible by num_heads"

        # Q K V 线性层
        self.query = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.embed_size, bias=False)

        # dropout
        self.attn_dropout = nn.Dropout(0.1)

        # 输出线性层
        self.output_linear = nn.Linear(self.head_dim * self.num_heads, self.embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 1. 初始化 Q K V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. 把 Q K V 拆为多头, [batch_size, seq_len, embed_size] -> [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. 交换顺序用于计算, [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        # 一次性计算所有的 batch 和 head, matmul 只会对最后两个维度做矩阵乘法
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(2, 3)) * self.head_dim**-0.5

        # 5. 应用 mask (如果有)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 6. 转换为概率分布
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 7. 应用 Dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 8. 计算输出
        attn_output = torch.matmul(attn_weights, V)

        # 9. 合并多头, [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, embed_size]
        # contiguous() 确保内存是连续的, 否则 view 会报错
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_size)
        )

        # 10. 输出线性层
        output = self.output_linear(attn_output)

        return output
