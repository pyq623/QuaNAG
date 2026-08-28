# __init__.py
"""
统一的数据加载模块
支持1D时序数据(HAR)和2D图像数据(Image)
"""

from .base_loader import (
    BaseDataLoader,
    DataLoaderFactory,
    get_data_loader
)

from .har_loader import (
    HARDataset,
    GenericHARDataset,
    HAR_DATASET_INFO,
    get_har_dataloaders
)

from .image_loader import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    MNISTDataset,
    IMAGE_DATASET_INFO,
    get_image_dataloaders
)

__all__ = [
    # 基类和工厂
    'BaseDataLoader',
    'DataLoaderFactory',
    'get_data_loader',
    
    # HAR相关
    'HARDataset',
    'GenericHARDataset',
    'HAR_DATASET_INFO',
    'get_har_dataloaders',
    
    # Image相关
    'CIFAR10Dataset',
    'CIFAR100Dataset',
    'MNISTDataset',
    'IMAGE_DATASET_INFO',
    'get_image_dataloaders',
]


def get_dataset_info(dataset_name: str, task_type: str = None):
    """
    获取数据集信息的便捷函数
    
    参数:
        dataset_name: 数据集名称
        task_type: 任务类型 ('har' 或 'image')，如果不指定则自动推断
    
    返回:
        数据集信息字典
    
    示例:
        >>> info = get_dataset_info('MMAct', task_type='har')
        >>> info = get_dataset_info('cifar10', task_type='image')
        >>> info = get_dataset_info('MMAct')  # 自动推断为HAR
    """
    if task_type is None:
        # 自动推断任务类型
        if dataset_name in HAR_DATASET_INFO:
            task_type = 'har'
        elif dataset_name.lower() in IMAGE_DATASET_INFO:
            task_type = 'image'
        else:
            raise ValueError(
                f"无法推断数据集 '{dataset_name}' 的任务类型，"
                f"请显式指定 task_type='har' 或 'image'"
            )
    
    # 获取对应的数据加载器
    loader = get_data_loader(task_type)
    return loader.get_dataset_info(dataset_name)


def create_calibration_loader(dataloader, num_batches: int = 5):
    """
    创建校准数据加载器的便捷函数（兼容旧代码）
    
    参数:
        dataloader: 原始数据加载器
        num_batches: 校准批次数
    
    返回:
        校准数据加载器
    """
    from torch.utils.data import DataLoader, Subset
    
    batch_size = dataloader.batch_size
    total_samples = num_batches * batch_size
    dataset = dataloader.dataset
    subset_indices = list(range(min(total_samples, len(dataset))))
    
    calibration_dataset = Subset(dataset, subset_indices)
    
    return DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=getattr(dataloader, 'num_workers', 0),
        pin_memory=getattr(dataloader, 'pin_memory', False)
    )


# 版本信息
__version__ = '1.0.0'
__author__ = 'Pupper Yard'
__description__ = 'Unified data loader for HAR and Image tasks'