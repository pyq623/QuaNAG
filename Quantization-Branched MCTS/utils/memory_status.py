import torch
from collections import OrderedDict
import torch.nn as nn
from typing import Tuple

def calculate_memory_usage(model: nn.Module, input_size: Tuple[int], device: torch.device = torch.device('cpu')) -> dict:
    """
    通过直接查询张量属性来稳健地计算模型的激活内存和参数内存。
    这个版本能够准确区分不同量化模式。
    """
    model = model.to(device)
    model.eval()
    activation_memory = 0
    parameter_memory = 0
    dummy_input = torch.randn(*input_size, device=device)

    hooks = []
    def forward_hook(module, input, output):
        nonlocal activation_memory
        # 直接查询输出张量的元素大小，这是最准确的方法
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output
        activation_memory += output_tensor.numel() * output_tensor.element_size()

    # 只在叶子模块上挂钩子， 以避免重复计算
    for layer in model.modules():
        if not list(layer.children()): 
            hooks.append(layer.register_forward_hook(forward_hook))

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    # 同样使用 element_size() 来计算参数内存
    for param in model.parameters():
        parameter_memory += param.numel() * param.element_size()

    activation_memory_MB = activation_memory / (1024 ** 2)
    parameter_memory_MB = parameter_memory / (1024 ** 2)

    return {
        "activation_memory_MB": activation_memory_MB,
        "parameter_memory_MB": parameter_memory_MB,
        "total_memory_MB": activation_memory_MB + parameter_memory_MB,
    }
