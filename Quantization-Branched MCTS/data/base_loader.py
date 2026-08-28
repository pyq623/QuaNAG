# base_loader.py
"""
统一的数据加载器抽象基类
支持1D时序数据(HAR)和2D图像数据(Image)的统一接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
from torch.utils.data import DataLoader


class BaseDataLoader(ABC):
    """数据加载器抽象基类"""
    
    def __init__(self, task_type: str):
        """
        初始化数据加载器
        
        参数:
            task_type: 任务类型 ('har' 或 'image')
        """
        self.task_type = task_type
    
    @abstractmethod
    def get_dataloaders(
        self, 
        root_dir: str, 
        batch_size: int, 
        datasets_list: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, DataLoader]]:
        """
        获取数据加载器
        
        参数:
            root_dir: 数据集根目录
            batch_size: 批次大小
            datasets_list: 数据集名称列表
            **kwargs: 其他参数 (num_workers, pin_memory等)
        
        返回:
            嵌套字典: {dataset_name: {'train': DataLoader, 'test': DataLoader}}
        """
        pass
    
    @abstractmethod
    def get_input_size(self, dataset_name: str, batch_size: int = 1) -> Tuple[int, ...]:
        """
        获取输入尺寸
        
        参数:
            dataset_name: 数据集名称
            batch_size: 批次大小
        
        返回:
            HAR: (batch_size, channels, time_steps)
            Image: (batch_size, channels, height, width)
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        获取数据集信息
        
        参数:
            dataset_name: 数据集名称
        
        返回:
            数据集信息字典
        """
        pass
    
    @abstractmethod
    def create_calibration_loader(
        self, 
        dataloader: DataLoader, 
        num_batches: int = 5
    ) -> DataLoader:
        """
        创建校准数据加载器(用于量化)
        
        参数:
            dataloader: 原始数据加载器
            num_batches: 校准批次数
        
        返回:
            校准数据加载器
        """
        pass
    
    def get_num_classes(self, dataset_name: str) -> int:
        """获取类别数"""
        info = self.get_dataset_info(dataset_name)
        return info.get('num_classes', 0)
    
    def get_input_channels(self, dataset_name: str) -> int:
        """获取输入通道数"""
        info = self.get_dataset_info(dataset_name)
        return info.get('channels', 0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task_type='{self.task_type}')"


class DataLoaderFactory:
    """数据加载器工厂类"""
    
    _loaders: Dict[str, type] = {}
    
    @classmethod
    def register(cls, task_type: str):
        """注册数据加载器的装饰器"""
        def wrapper(loader_class):
            cls._loaders[task_type.lower()] = loader_class
            return loader_class
        return wrapper
    
    @classmethod
    def create(cls, task_type: str, **kwargs) -> BaseDataLoader:
        """
        创建数据加载器实例
        
        参数:
            task_type: 任务类型 ('har' 或 'image')
            **kwargs: 传递给数据加载器的额外参数
        
        返回:
            数据加载器实例
        
        示例:
            >>> loader = DataLoaderFactory.create('har')
            >>> loader = DataLoaderFactory.create('image')
        """
        task_type_lower = task_type.lower()
        
        if task_type_lower not in cls._loaders:
            available = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"Unknown task_type: '{task_type}'. "
                f"Available types: {available}"
            )
        
        loader_class = cls._loaders[task_type_lower]
        return loader_class(task_type=task_type_lower, **kwargs)
    
    @classmethod
    def list_available_types(cls) -> List[str]:
        """列出所有可用的任务类型"""
        return list(cls._loaders.keys())


def get_data_loader(task_type: str, **kwargs) -> BaseDataLoader:
    """
    便捷函数：获取数据加载器
    
    参数:
        task_type: 任务类型 ('har' 或 'image')
        **kwargs: 传递给数据加载器的额外参数
    
    返回:
        数据加载器实例
    
    示例:
        >>> har_loader = get_data_loader('har')
        >>> image_loader = get_data_loader('image')
    """
    return DataLoaderFactory.create(task_type, **kwargs)