"""
EdgeGen2.0 配置文件模块

提供以下配置文件的统一访问接口:
- llm_config.yaml       : LLM服务配置
- search_space.yaml     : 架构搜索空间定义
- mapping.yaml          : NCode编码映射
"""

from pathlib import Path
import yaml
from typing import Dict, Any

# 获取configs目录路径
_CONFIG_DIR = Path(__file__).parent

# 缓存加载的配置
_config_cache: Dict[str, Any] = {}

def _load_config(file_name: str) -> Dict[str, Any]:
    """加载指定的YAML配置文件"""
    if file_name not in _config_cache:
        config_path = _CONFIG_DIR / file_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            _config_cache[file_name] = yaml.safe_load(f)
    return _config_cache[file_name]

# 为每个配置文件创建访问函数
def get_llm_config() -> Dict[str, Any]:
    """获取LLM服务配置"""
    return _load_config("llm_config.yaml")


def get_search_space() -> Dict[str, Any]:
    """获取架构搜索空间定义"""
    return _load_config("search_space.yaml")

def get_search_space_none() -> Dict[str, Any]:
    """获取无量化搜索空间定义"""
    return _load_config("search_space_none.yaml")

def get_mapping() -> Dict[str, Any]:
    """获取NCode映射配置"""
    return _load_config("search_space_mapping.yaml")


def resolve_channels(channel_code: str, input_channels: int) -> int:
    """
    解析channels代码为实际通道数
    
    Args:
        channel_code: 单个数字字符 "0"-"4"
        input_channels: 数据集的输入通道数
        
    Returns:
        实际通道数
    """
    mapping = get_mapping()
    channel_value = mapping['reverse_mapping']['channels'][int(channel_code)]
    
    if channel_value == "input":
        return input_channels
    return channel_value

# 显式导出列表
__all__ = [
    'get_llm_config',
    'get_search_space',
    'get_search_space_none',
    'get_mapping',
    'resolve_channels'
]

# 版本信息
__version__ = '0.1.0'


# 示例用法
# from tinyml.configs import (
#     get_llm_config,
#     get_search_space
# )

# # 获取配置
# llm_config = get_llm_config()
# search_space = get_search_space()

# # 在代码中使用
# print("LLM配置:", llm_config['model_name'])
# print("搜索空间约束:", search_space['constraints'])

# from configs import get_mapping, resolve_channels

# # 加载映射
# mapping = get_mapping()

# 解析NCode: "1|11010001-4|11021201-41010120-0|42121120-1"
# 假设input_channels = 23 (Mhealth数据集)

# 第一个stage的channels是 "4" -> resolve_channels("4", 23) -> 23
# 第二个stage的channels是 "0" -> resolve_channels("0", 23) -> 6
# 第三个stage的channels是 "1" -> resolve_channels("1", 23) -> 12