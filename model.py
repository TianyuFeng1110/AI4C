import torch
import torch.nn as nn
import math

# 位置编码模块保持不变
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, coord_dim=2, freq_bands=64):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Linear(2*2*freq_bands, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.freq_bands = freq_bands
        self.register_buffer(
            'frequencies', 
            2.0 ** torch.linspace(0, 9, freq_bands)[None, None, :]
        )

    def _fourier_mapping(self, coords):
        scaled_coords = coords * 2 - 1
        scaled_coords = scaled_coords.unsqueeze(-1)
        angles = scaled_coords * self.frequencies * math.pi
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat([sin, cos], dim=-1).flatten(start_dim=1)

    def forward(self, coords):
        fourier = self._fourier_mapping(coords)
        return self.coord_proj(fourier)

# 注意力层保持兼容性
class AttentionAwareTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        self.attention_weights = attn_weights
        return self.dropout1(x)

class MultiTaskGridTransformer(nn.Module):
    def __init__(self, input_size, d_model=512, 
                 light_heads=2, light_layers=2,
                 temp_heads=2, temp_layers=2,
                 walk_heads=4, walk_layers=3,
                 dropout=0.3):
        """
        多任务 Grid Transformer 模型
        参数说明：
            input_size: 输入特征维度
            d_model: 基础模型维度
            light_*: 光污染分支参数 (1个输出)
            temp_*: 热环境分支参数 (2个输出)
            walk_*: 慢行环境分支参数 (5个输出)
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.record_attention = False
        
        # 特征投影层共享
        self.feature_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码共享
        self.position_encoding = PositionalEncoding(d_model)
        
        # 三个独立编码器分支（保持原结构）
        self.light_encoder = self._build_encoder(light_heads, light_layers)
        self.temp_encoder = self._build_encoder(temp_heads, temp_layers)
        self.walk_encoder = self._build_encoder(walk_heads, walk_layers)
        
        # 独立输出头（每个预测指标一个MLP）
        self.output_heads = nn.ModuleDict({
            'light': self._build_output_head(1),  # 光污染：1个指标
            
            'temp_1': self._build_output_head(1), # 热环境：2个指标
            'temp_2': self._build_output_head(1),
            
            'walk_1': self._build_output_head(1), # 慢行环境：5个指标
            'walk_2': self._build_output_head(1),
            'walk_3': self._build_output_head(1),
            'walk_4': self._build_output_head(1),
            'walk_5': self._build_output_head(1)
        })

        # 注意力矩阵存储（保持原结构）
        self.attention_maps = {
            'light': [], 'temp': [], 'walk': []
        }
        
        # 初始化权重（保持原结构）
        self._init_weights()

    def _build_output_head(self, output_dim):
        """构建独立输出头"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model//2, output_dim)
        )
    
    def _build_encoder(self, num_heads, num_layers):
        """构建独立编码器分支"""
        return nn.ModuleList([
            AttentionAwareTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=self.d_model*4,
                dropout=0.25,
                batch_first=False,
                norm_first=True
            ) for _ in range(num_layers)
        ])

    def _init_weights(self):
        """权重初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def enable_attention_recording(self, enable=True):
        self.record_attention = enable

    def _process_branch(self, x, encoder, output_layer):
        """处理单个任务分支"""
        attention_weights = []
        
        # Transformer 编码
        for layer in encoder:
            x = layer(x)
            if self.record_attention:
                attention_weights.append(layer.attention_weights)
        
        # 输出处理
        outputs = output_layer(x)
        
        return outputs, attention_weights

    def forward(self, features, coordinates):
        """
        前向传播
        输入:
            features: [seq_len, input_size]
            coordinates: [seq_len, 2]
        输出:
            dict: 包含三个任务的预测结果
                - light: [s, 1]
                - temp: [s, 2]
                - walk: [s, 5]
        """
        # 特征编码阶段
        x = self.feature_proj(features)  # [s, d_model]
        pos_embed = self.position_encoding(coordinates)
        x = x + pos_embed
        x = x.unsqueeze(1)  # [s, 1, d_model]

        # 分支处理
        light_out, light_attn = self._process_branch(
            x, self.light_encoder, lambda x: x
        )
        temp_out, temp_attn = self._process_branch(
            x, self.temp_encoder, lambda x: x
        )
        walk_out, walk_attn = self._process_branch(
            x, self.walk_encoder, lambda x: x
        )

        # 应用独立输出头并拼接
        light_pred = self.output_heads['light'](light_out).squeeze(1)  # [s, 1]

        temp_pred = torch.cat([
            self.output_heads['temp_1'](temp_out),  # [s, 1, 1]
            self.output_heads['temp_2'](temp_out)
        ], dim=2).squeeze(1)  # [s, 2]

        walk_pred = torch.cat([
            self.output_heads['walk_1'](walk_out),
            self.output_heads['walk_2'](walk_out),
            self.output_heads['walk_3'](walk_out),
            self.output_heads['walk_4'](walk_out),
            self.output_heads['walk_5'](walk_out)
        ], dim=2).squeeze(1)  # [s, 5]

        # 保存注意力
        if self.record_attention:
            self.attention_maps['light'] = light_attn
            self.attention_maps['temp'] = temp_attn
            self.attention_maps['walk'] = walk_attn

        return {
            'light': light_pred,
            'temp': temp_pred,
            'walk': walk_pred
        }
    
    def get_attention_maps(self, branch=None):
        """获取注意力矩阵"""
        if branch is None:
            return self.attention_maps
        return self.attention_maps.get(branch, [])

    def clear_attention_maps(self):
        self.attention_maps = {
            'light': [], 'temp': [], 'walk': []
        }