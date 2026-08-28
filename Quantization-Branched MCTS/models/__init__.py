# /root/unified_nas/models/__init__.py
"""
统一的模型模块
支持1D时序数据(HAR)和2D图像数据(Image)
"""

from .base_model import TinyMLModel
from .candidate_models import CandidateModel
from .conv_blocks import (
    DWSepConvBlock,
    MBConvBlock,
    DpConvBlock,
    SeSepConvBlock,
    SeDpConvBlock,
    SEBlock,
    get_activation,
    get_conv_layer,
    get_batchnorm_layer,
    get_adaptive_avgpool_layer
)
from .QuantizableModel import (
    QuantizableModel,
    evaluate_quantized_model,
    apply_configurable_static_quantization,
    fuse_model_modules,
    get_static_quantization_config,
    apply_qaft_quantization,
    prepare_qaft_model,
    fuse_QATmodel_modules,
    get_quantization_option
)

__all__ = [
    # 基础模型
    'TinyMLModel',
    'CandidateModel',
    
    # 卷积块
    'DWSepConvBlock',
    'MBConvBlock',
    'DpConvBlock',
    'SeSepConvBlock',
    'SeDpConvBlock',
    'SEBlock',
    
    # 辅助函数
    'get_activation',
    'get_conv_layer',
    'get_batchnorm_layer',
    'get_adaptive_avgpool_layer',
    
    # 量化相关
    'QuantizableModel',
    'evaluate_quantized_model',
    'apply_configurable_static_quantization',
    'fuse_model_modules',
    'get_static_quantization_config',
    'apply_qaft_quantization',
    'prepare_qaft_model',
    'fuse_QATmodel_modules',
    'get_quantization_option'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Pupper Yard'
__description__ = 'Unified TinyML models for HAR and Image tasks'