# /root/unified_nas/models/base_model.py
"""
统一的TinyML模型基类
支持1D时序数据(HAR)和2D图像数据(Image)
"""
import torch
import torch.nn as nn
from .conv_blocks import (
    DWSepConvBlock, MBConvBlock, get_activation, 
    DpConvBlock, SeDpConvBlock, SeSepConvBlock,
    get_adaptive_avgpool_layer
)
import numpy as np
from torch.quantization import QuantStub, DeQuantStub

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class TinyMLModel(nn.Module):
    """
    统一的TinyML模型
    
    通过task_type参数自动切换1D/2D操作:
    - task_type='har': 使用Conv1d, BatchNorm1d, AdaptiveAvgPool1d
    - task_type='image': 使用Conv2d, BatchNorm2d, AdaptiveAvgPool2d
    """
    def __init__(self, config, task_type='har'):
        """
        初始化模型
        
        参数:
            config: 模型配置字典
            task_type: 任务类型 ('har' 或 'image')
        """
        super().__init__()
        self.config = config
        self.task_type = task_type
        
        # 如果启用量化模式，初始化量化模块
        self.use_quant = self.config.get("quant_mode", None) is not None
        if self.use_quant:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        
        self.stages = nn.ModuleList()
        
        # 构建模型
        final_in_channels = self._build_model()
        
        # 根据task_type选择池化层
        AdaptiveAvgPool = get_adaptive_avgpool_layer(task_type)
        self.avgpool = AdaptiveAvgPool(1)
        
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(
            final_in_channels, 
            self.config.get("num_classes", 10)
        )
        
        # 输出维度
        self.output_dim = self.config.get("num_classes", 10)
        # ✅ 修改：output_dim 是 classifier 之前的特征维度
        # self.output_dim = final_in_channels

    def _build_model(self):
        """构建模型各个stage"""
        # HAR默认6通道，Image默认3通道
        default_channels = 6 if self.task_type == 'har' else 3
        in_channels = self.config.get("input_channels", default_channels)
        quant_mode = self.config.get("quant_mode", None)
        
        current_in_channels = in_channels
        
        for stage_config in self.config["stages"]:
            out_channels = stage_config["channels"]
            blocks = []
            
            for block_config in stage_config["blocks"]:
                block_type = block_config["type"]
                
                # 所有block都传入task_type参数
                if block_type == "DWSepConv":
                    block = DWSepConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", False),
                        se_ratio=block_config.get("se_ratios", 0),
                        activation=block_config["activation"],
                        skip_connection=block_config.get("skip_connection", True),
                        quant_mode=quant_mode,
                        task_type=self.task_type  # ✅ 传入task_type
                    )
                    
                elif block_type == "MBConv":
                    block = MBConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        expansion=block_config["expansion"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", False),
                        se_ratio=block_config.get("se_ratios", 0),
                        activation=block_config["activation"],
                        skip_connection=block_config.get("skip_connection", True),
                        quant_mode=quant_mode,
                        task_type=self.task_type  # ✅ 传入task_type
                    )

                elif block_type == "DpConv":
                    block = DpConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        activation=block_config.get("activation", "ReLU6"),
                        quant_mode=quant_mode,
                        task_type=self.task_type  # ✅ 传入task_type
                    )
                    
                elif block_type == "SeSepConv":
                    block = SeSepConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", True),
                        se_ratio=block_config.get("se_ratio", 0.25),
                        activation=block_config.get("activation", "ReLU6"),
                        se_activation=block_config.get("se_activation", "Sigmoid"),
                        quant_mode=quant_mode,
                        task_type=self.task_type  # ✅ 传入task_type
                    )
                    
                elif block_type == "SeDpConv":
                    block = SeDpConvBlock(
                        in_channels=current_in_channels,
                        out_channels=out_channels,
                        kernel_size=block_config["kernel_size"],
                        stride=block_config.get("stride", 1),
                        has_se=block_config.get("has_se", True),
                        se_ratio=block_config.get("se_ratio", 0.25),
                        activation=block_config.get("activation", "ReLU6"),
                        se_activation=block_config.get("se_activation", "Sigmoid"),
                        quant_mode=quant_mode,
                        task_type=self.task_type  # ✅ 传入task_type
                    )

                else:
                    raise ValueError(f"Unknown block type: {block_type}")
                
                blocks.append(block)
                current_in_channels = out_channels
            
            stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            
        return current_in_channels
        
    def forward(self, x):
        """前向传播"""
        # 如果启用量化，执行量化操作
        if self.use_quant:
            x = self.quant(x)

        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        
        x = self.classifier(x)
        
        # 如果启用量化，执行反量化操作
        if self.use_quant:
            x = self.dequant(x)
            
        return x