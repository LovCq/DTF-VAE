import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# 包含多头自注意力和前馈网络的结构
class EncoderLayer_selfattn(nn.Module):
    """Compose with two layers"""
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(EncoderLayer_selfattn, self).__init__()
        # 多头自注意力模块
        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            dropout=dropout
        )
        # 前馈神经网络模块
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        # 自注意力计算（输入为 Q=K=V=enc_input）
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        # 前馈网络处理
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn  # 返回输出和注意力权重

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.n_head = n_head  # 注意力头数
        self.d_model = d_model

        # 动态计算特征维度
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.d_k = d_model // n_head  # ✅ 自动计算
        self.d_v = d_model // n_head  # ✅ 自动计算

        # 初始化线性变换层
        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)

        # 参数初始化（He初始化变种）
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        # 缩放点积注意力模块
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        # 输出线性变换
        self.fc = nn.Linear(n_head * self.d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)  # Xavier初始化
        self.dropout = nn.Dropout(dropout)      # 随机失活

    def forward(self, q, k, v):
        # 输入维度: [batch_size, seq_len, d_model]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q  # 残差连接保留原始输入
        # 线性变换并分割为多个头
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # 维度调整（重点修改部分）
        # 原错误代码：q.permute(2, 0, 1, 3) → 错误维度顺序
        q = q.permute(0, 2, 1, 3).contiguous().view(-1, len_q, d_k)  # [batch*head, len_q, d_k]
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, len_k, d_k)  # [batch*head, len_k, d_k]
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, len_v, d_v)  # [batch*head, len_v, d_v]

        # 新增维度验证（确保Q/K/V一致性）
        assert q.size(0) == k.size(0) == v.size(0), f"头数不一致 Q:{q.size(0)} K:{k.size(0)} V:{v.size(0)}"
        assert q.size(-1) == d_k, f"Q特征维度错误，预期{d_k}，实际{q.size(-1)}"
        assert k.size(-1) == d_k, f"K特征维度错误，预期{d_k}，实际{k.size(-1)}"
        assert v.size(-1) == d_v, f"V特征维度错误，预期{d_v}，实际{v.size(-1)}"
        assert self.d_k == self.d_model // self.n_head, \
            f"特征维度计算错误: d_k={self.d_k} ≠ d_model//n_head={self.d_model // self.n_head}"
        # 计算缩放点积注意力
        output, attn = self.attention(q, k, v)
        # 合并多头输出：[B, L, H*d_v]
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        # 线性变换 + 残差连接 + 层归一化
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

# 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)  # 沿序列维度做Softmax

    def forward(self, q, k, v):
        # 计算注意力得分: [H*B, L, L]
        attn = torch.bmm(q, k.transpose(1, 2))  # 批量矩阵乘法
        attn = attn / self.temperature          # 缩放
        # 应用Softmax和Dropout
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # 加权求和
        output = torch.bmm(attn, v)
        return output, attn

# 位置式前馈网络（两层卷积）
class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 第一层卷积：扩展维度
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # 1x1卷积
        # 第二层卷积：降维回原始维度
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)  # 层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x  # 残差连接保留输入
        # 调整维度: [B, L, d_in] → [B, d_in, L]
        output = x.transpose(1, 2)
        # 第一层卷积 + ReLU激活
        output = self.w_2(F.relu(self.w_1(output)))
        # 恢复维度: [B, d_in, L] → [B, L, d_in]
        output = output.transpose(1, 2)
        # 随机失活 + 残差连接 + 层归一化
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
