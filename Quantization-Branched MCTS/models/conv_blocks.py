# /root/unified_nas/models/conv_blocks.py
"""
统一的卷积块实现
支持1D时序数据(HAR)和2D图像数据(Image)
"""
import torch
import torch.nn as nn
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def get_activation(activation_name, quant_mode=None):
    """
    获取激活函数
    
    参数:
        activation_name: 激活函数名称
        quant_mode: 量化模式（如果启用量化，Swish会被替换为Hardswish）
    """
    if quant_mode is not None and activation_name == 'Swish':
        return nn.Hardswish()
    
    activations = {
        'ReLU': lambda: nn.ReLU(inplace=True),
        'ReLU6': lambda: nn.ReLU6(),
        'LeakyReLU': lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
        'Swish': lambda: nn.SiLU(),
        'Sigmoid': lambda: nn.Sigmoid(),
        'HardSigmoid': lambda: nn.Hardsigmoid(),
        'Hardswish': lambda: nn.Hardswish()
    }
    return activations.get(activation_name, lambda: nn.ReLU(inplace=True))()


def get_conv_layer(task_type='har'):
    """根据任务类型返回对应的卷积层"""
    if task_type == 'har':
        return nn.Conv1d
    elif task_type == 'image':
        return nn.Conv2d
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def get_batchnorm_layer(task_type='har'):
    """根据任务类型返回对应的BatchNorm层"""
    if task_type == 'har':
        return nn.BatchNorm1d
    elif task_type == 'image':
        return nn.BatchNorm2d
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def get_adaptive_avgpool_layer(task_type='har'):
    """根据任务类型返回对应的自适应平均池化层"""
    if task_type == 'har':
        return nn.AdaptiveAvgPool1d
    elif task_type == 'image':
        return nn.AdaptiveAvgPool2d
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation块
    自动适配1D和2D
    """
    def __init__(self, channel, se_ratio, se_activation='Sigmoid', 
                 quant_mode=None, task_type='har'):
        super().__init__()
        self.task_type = task_type
        
        reduced_ch = max(1, int(channel * se_ratio))
        
        # 根据 task_type 选择对应的层
        Conv = get_conv_layer(task_type)
        AdaptiveAvgPool = get_adaptive_avgpool_layer(task_type)
        
        self.se = nn.Sequential(
            AdaptiveAvgPool(1),
            Conv(channel, reduced_ch, 1),
            nn.ReLU(inplace=True),
            Conv(reduced_ch, channel, 1),
            get_activation(se_activation, quant_mode)
        )
        self.mul_op = nn.quantized.FloatFunctional()
    
    def forward(self, x):
        se_out = self.se(x)
        return self.mul_op.mul(x, se_out)


class DWSepConvBlock(nn.Module):
    """
    Depthwise Separable卷积块
    自动适配1D和2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, has_se=False, se_ratio=0.25, 
                 activation='ReLU6', skip_connection=True, 
                 quant_mode=None, task_type='har'):
        super().__init__()
        self.task_type = task_type
        
        # 根据task_type选择对应的层
        Conv = get_conv_layer(task_type)
        BatchNorm = get_batchnorm_layer(task_type)
        
        self.dw_conv = nn.Sequential(
            Conv(in_channels, in_channels, kernel_size, 
                 stride=stride, padding=kernel_size//2, groups=in_channels),
            BatchNorm(in_channels),
            get_activation(activation, quant_mode)
        )
        
        self.pw_conv = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            BatchNorm(out_channels)
        )
        
        self.skip_connection = skip_connection and (stride == 1) and (in_channels == out_channels)
        if self.skip_connection:
            self.skip_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        identity = x
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        if self.skip_connection:
            return self.skip_add.add(out, identity)
        return out


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck卷积块
    自动适配1D和2D
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 expansion=4, stride=1, has_se=False, se_ratio=0.25, 
                 activation='ReLU6', skip_connection=True, 
                 quant_mode=None, se_activation='Sigmoid', task_type='har'):
        super().__init__()
        self.task_type = task_type
        hidden_dim = int(in_channels * expansion)
        
        # 根据task_type选择对应的层
        Conv = get_conv_layer(task_type)
        BatchNorm = get_batchnorm_layer(task_type)
        
        self.use_expand = expansion != 1
        if self.use_expand:
            self.expand_conv = nn.Sequential(
                Conv(in_channels, hidden_dim, 1),
                BatchNorm(hidden_dim),
                get_activation(activation, quant_mode)
            )
        
        self.dw_conv = nn.Sequential(
            Conv(hidden_dim, hidden_dim, kernel_size, 
                 stride=stride, padding=kernel_size//2, groups=hidden_dim),
            BatchNorm(hidden_dim),
            get_activation(activation, quant_mode)
        )
        
        self.se = SEBlock(hidden_dim, se_ratio, se_activation, 
                         quant_mode, task_type) if has_se and se_ratio > 0 else None
        
        self.pw_conv = nn.Sequential(
            Conv(hidden_dim, out_channels, 1),
            BatchNorm(out_channels)
        )
        
        self.skip_connection = skip_connection and (stride == 1) and (in_channels == out_channels)
        if self.skip_connection:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.expand_conv(x) if self.use_expand else x
        out = self.dw_conv(out)
        if self.se is not None:
            out = self.se(out)
        out = self.pw_conv(out)
        if self.skip_connection:
            return self.skip_add.add(out, identity)
        return out


class DpConvBlock(nn.Module):
    """
    纯Depthwise + Pointwise卷积（无SE模块）
    自动适配1D和2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, activation='ReLU6', quant_mode=None, task_type='har'):
        super().__init__()
        self.task_type = task_type
        
        # 根据task_type选择对应的层
        Conv = get_conv_layer(task_type)
        BatchNorm = get_batchnorm_layer(task_type)
        
        self.dw_conv = nn.Sequential(
            Conv(in_channels, in_channels, kernel_size, 
                 stride=stride, padding=kernel_size//2, groups=in_channels),
            BatchNorm(in_channels),
            get_activation(activation, quant_mode)
        )
        
        self.pw_conv = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            BatchNorm(out_channels)
        )
        
    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        return out


class SeSepConvBlock(nn.Module):
    """
    带SE模块的Depthwise Separable卷积
    自动适配1D和2D
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, has_se=True, se_ratio=0.25, 
                 activation='ReLU6', se_activation='Sigmoid', 
                 quant_mode=None, task_type='har'):
        super().__init__()
        self.task_type = task_type
        
        # 根据task_type选择对应的层
        Conv = get_conv_layer(task_type)
        BatchNorm = get_batchnorm_layer(task_type)
        
        self.dw_conv = nn.Sequential(
            Conv(in_channels, in_channels, kernel_size, 
                 stride=stride, padding=kernel_size//2, groups=in_channels),
            BatchNorm(in_channels),
            get_activation(activation, quant_mode)
        )
        
        self.se = SEBlock(in_channels, se_ratio, se_activation, 
                         quant_mode, task_type) if has_se and se_ratio > 0 else None
        
        self.pw_conv = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            BatchNorm(out_channels)
        )
        
    def forward(self, x):
        out = self.dw_conv(x)
        if self.se is not None:
            out = self.se(out)
        out = self.pw_conv(out)
        return out


class SeDpConvBlock(nn.Module):
    """
    带SE模块的纯Depthwise卷积 （无 Pointwise ） 
    自动适配1D和2D
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, has_se=True, se_ratio=0.25, 
                 activation='ReLU6', se_activation='Sigmoid', 
                 quant_mode=None, task_type='har'):
        super().__init__()
        self.task_type = task_type
        
        # 根据task_type选择对应的层
        Conv = get_conv_layer(task_type)
        BatchNorm = get_batchnorm_layer(task_type)
        
        self.dw_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, 
                 stride=stride, padding=kernel_size//2, groups=in_channels),
            BatchNorm(out_channels),
            get_activation(activation, quant_mode)
        )
        
        self.se = SEBlock(out_channels, se_ratio, se_activation, 
                         quant_mode, task_type) if has_se and se_ratio > 0 else None

    def forward(self, x):
        out = self.dw_conv(x)
        if self.se is not None:
            out = self.se(out)
        return out