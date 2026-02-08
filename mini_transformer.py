"""
项目：基于轻量级Transformer的手写文本序列识别
文件：mini_transformer.py
作者：林泽远
日期：2026.02
功能：实现适配MNIST的轻量级Transformer，仅保留核心Self-Attention/Encoder/位置编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ====================== 配置类（统一管理参数，科研项目必备） ======================
class MiniTransformerConfig:
    def __init__(
        self,
        img_size: int = 28,          # MNIST图片尺寸
        in_channels: int = 1,        # 输入通道数（灰度图=1）
        num_classes: int = 10,       # 分类数（MNIST=10）
        embed_dim: int = 64,         # 嵌入维度（轻量化，不用大）
        num_heads: int = 2,          # 注意力头数（大二先设2）
        num_layers: int = 1,         # Encoder层数（轻量化）
        max_seq_len: int = 784       # 序列长度（28*28=784）
    ):
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.head_dim = embed_dim // num_heads  # 每个头的维度

# ====================== 核心模块1：自注意力（Self-Attention） ======================
class SelfAttention(nn.Module):
    def __init__(self, config: MiniTransformerConfig):
        super().__init__()
        self.config = config

        # Q/K/V 线性投影层
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状 [batch_size, seq_len, embed_dim]
        Returns:
            注意力输出，形状 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 1. 生成Q/K/V，并拆分注意力头
        q = self.q_proj(x).reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)

        # 2. 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.config.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)  # 注意力权重归一化

        # 3. 加权求和 + 拼接注意力头
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, self.config.embed_dim)
        
        # 4. 输出投影
        output = self.out_proj(attn_output)
        return output

# ====================== 核心模块2：Transformer Encoder层 ======================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: MiniTransformerConfig):
        super().__init__()
        self.config = config
        self.attn = SelfAttention(config)  # 自注意力层
        self.ffn = nn.Sequential(          # 前馈网络（轻量化版）
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim)
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)  # 层归一化（防止梯度爆炸）
        self.norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（带残差连接）"""
        # 自注意力 + 残差 + 归一化
        x = self.norm1(x + self.attn(x))
        # 前馈网络 + 残差 + 归一化
        x = self.norm2(x + self.ffn(x))
        return x

# ====================== 核心模块3：位置编码（Transformer必备） ======================
class PositionalEncoding(nn.Module):
    def __init__(self, config: MiniTransformerConfig):
        super().__init__()
        self.config = config

        # 生成位置编码矩阵（固定值，不参与训练）
        pe = torch.zeros(config.max_seq_len, config.embed_dim)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / config.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 注册为非训练参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        x = x + self.pe[:x.size(1), :]
        return x

# ====================== 最终：Mini Transformer（适配MNIST） ======================
class MiniTransformer(nn.Module):
    def __init__(self, config: Optional[MiniTransformerConfig] = None):
        super().__init__()
        # 默认配置（没传就用这个）
        self.config = config if config is not None else MiniTransformerConfig()
        
        # 1. 图片→序列投影（28*28*1 → embed_dim）
        self.img2seq = nn.Linear(
            self.config.in_channels * self.config.img_size * self.config.img_size,
            self.config.embed_dim
        )
        
        # 2. 位置编码
        self.pos_enc = PositionalEncoding(self.config)
        
        # 3. Transformer Encoder（多层堆叠）
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(self.config) for _ in range(self.config.num_layers)
        ])
        
        # 4. 分类头（序列→分类结果）
        self.classifier = nn.Linear(self.config.embed_dim, self.config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size = x.shape[0]
            
        # 🔥 修复：先展平→投影（784→64），再reshape成序列
        # 1. 展平图片：[batch_size, 784]
        x = x.flatten(1)
        # 2. 图片投影到嵌入维度：[batch_size, 64]
        x = self.img2seq(x)
        # 3. reshape成序列：[batch_size, 1, 64]（seq_len=1，因为是单张图片分类）
        x = x.reshape(batch_size, 1, self.config.embed_dim)

        # 4. 位置编码（seq_len=1，编码不影响，但保留逻辑）
        x = self.pos_enc(x)

        # 5. Transformer编码
        x = self.encoder(x)

        # 6. 序列均值池化 + 分类
        x = x.mean(dim=1)  # 简单池化，大二够用
        output = self.classifier(x)

        return output

# ====================== 测试代码（验证模型可运行） ======================
if __name__ == "__main__":
    # 1. 初始化配置
    config = MiniTransformerConfig(
        img_size=28,
        embed_dim=64,
        num_heads=2,
        num_layers=1
    )
    
    # 2. 初始化模型
    model = MiniTransformer(config)
    
    # 3. 测试输入（MNIST样例）
    dummy_input = torch.randn(4, 1, 28, 28)  # batch_size=4
    output = model(dummy_input)
    
    # 4. 打印关键信息（科研调试必备）
    print(f"✅ 模型初始化成功！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape} (预期: [4, 10])")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1000:.2f}k")