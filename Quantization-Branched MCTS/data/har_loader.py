# har_loader.py
"""
1D时序HAR数据加载器
"""
from typing import Dict, Any, Optional, List
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from .base_loader import BaseDataLoader, DataLoaderFactory

class AddGaussianNoise:
    """
    可序列化的高斯噪声数据增强类
    替代 lambda 函数以支持多进程 DataLoader
    """
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        return x + torch.randn_like(x) * self.std
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"

# HAR数据集信息配置
HAR_DATASET_INFO = {
    'USCHAD': {
        'task_type': 'har',  # ✅ 添加
        'channels': 6,
        'time_steps': 1000,
        'num_classes': 12,
        'description': 'A group of sensing data based on the right front hip, including walking, running, going upstairs, going downstairs, jumping, sitting, standing, sleeping and taking the elevator.'
    },
    'UTD-MHAD': {
        'task_type': 'har',  # ✅ 添加
        'channels': 6,
        'time_steps': 300,
        'num_classes': 27,
        'description': 'A group of sensing data based on the right wrist or right thigh, including waving, punching, clapping, jumping, push ups and other actions.'
    },
    'Wharf': {
        'task_type': 'har',  # ✅ 添加
        'channels': 3,
        'time_steps': 320,
        'num_classes': 14,
        'description': 'The wrist-based HAR dataset includes loading/unloading, forklift operation, and walking activities (and so on).'
    },
    'Mhealth': {
        'task_type': 'har',  # ✅ 添加
        'channels': 23,
        'time_steps': 500,
        'num_classes': 12,
        'description': 'The chest-and-wrist HAR dataset includes standing, sitting, supine, and lateral lying postures (and so on).'
    },
    'MMAct': {
        'task_type': 'har',  # ✅ 添加
        'channels': 9,
        'time_steps': 250,
        'num_classes': 35,
        'description': 'The pants-pocket HAR dataset includes walking, running, and cycling activities (and so on).'
    }
}


class HARDataset(Dataset):
    """HAR70+ Dataset Loader"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'har70plus')
        self.split = split
        self.transform = transform
        
        # Load data and labels
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))
        
        # Load label dictionary
        with open(os.path.join(self.root_dir, 'har70plus.json'), 'r') as f:
            self.label_dict = json.load(f)['label_dictionary']
        
        # Convert to PyTorch tensors
        self.X = torch.from_numpy(self.X).float()  # [N, 500, 6]
        self.y = torch.from_numpy(self.y).long()

        # Add classes attribute from label_dict
        self.classes = list(self.label_dict.values())
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # [500, 6]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x.permute(1, 0), y  # [6, 500]


class GenericHARDataset(Dataset):
    """通用HAR数据集加载器"""
    def __init__(self, root_dir, dataset_name, split='train', transform=None):
        """
        初始化通用数据集加载器

        参数:
            root_dir (str): 数据集根目录
            dataset_name (str): 数据集名称
            split (str): 数据集划分（'train' 或 'test'）
            transform (callable, optional): 数据变换
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        self.transform = transform
        
        # 加载数据和标签
        self.X = np.load(os.path.join(self.root_dir, f'X_{split}.npy'))
        self.y = np.load(os.path.join(self.root_dir, f'y_{split}.npy'))

        # 确保它们是标准的 numpy 数组
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        
        # 添加类型断言（可选，用于调试）
        assert isinstance(self.X, np.ndarray), f"X must be np.ndarray, got {type(self.X)}"
        assert isinstance(self.y, np.ndarray), f"y must be np.ndarray, got {type(self.y)}"

        # 加载标签字典（如果存在）
        label_dict_path = os.path.join(self.root_dir, f'{dataset_name}.json')
        if os.path.exists(label_dict_path):
            with open(label_dict_path, 'r') as f:
                self.label_dict = json.load(f).get('label_dictionary', {})
            self.classes = list(self.label_dict.values())
        else:
            # 如果没有 JSON 文件，则用数字类
            self.label_dict = None
            self.classes = list(range(int(self.y.max() + 1)))
        
        # 转换为 PyTorch 张量
        self.X = torch.FloatTensor(self.X)  # 使用FloatTensor更明确
        self.y = torch.LongTensor(self.y)  # 使用LongTensor更明确
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # 转换为 [C, T] 格式
        return x.permute(1, 0), y  # 将 [T, C] 转换为 [C, T]


@DataLoaderFactory.register('har')
class HARDataLoader(BaseDataLoader):
    """HAR数据加载器实现"""
    
    def __init__(self, task_type: str = 'har'):
        super().__init__(task_type)
        self.dataset_info_map = HAR_DATASET_INFO
    
    def get_dataloaders(
        self, 
        root_dir: str, 
        batch_size: int = 64, 
        datasets_list: Optional[List[str]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, DataLoader]]:
        """
        获取HAR数据加载器
        
        参数:
            root_dir: 数据集根目录
            batch_size: 批次大小
            datasets_list: 数据集名称列表
            num_workers: 数据加载工作进程数
            pin_memory: 是否固定内存
        
        返回:
            {dataset_name: {'train': DataLoader, 'test': DataLoader}}
        """
        # # 数据增强：添加轻微噪声
        # transform = transforms.Compose([
        #     transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        # ])

        # ✅ 使用可序列化的类替代 lambda 函数
        transform = transforms.Compose([
            AddGaussianNoise(std=0.01),  # 添加轻微高斯噪声
        ])
        
        # 默认加载所有数据集
        if datasets_list is None:
            datasets_list = ['DSADS', 'har70plus', 'Harth', 'Mhealth', 'MMAct', 
                            'MotionSense', 'Opp_g', 'PAMAP', 'realworld', 'Shoaib', 
                            'TNDA-HAR', 'UCIHAR', 'USCHAD', 'ut-complex', 'UTD-MHAD', 
                            'w-HAR', 'Wharf', 'WISDM']
        
        dataloaders = {}
        
        for dataset_name in datasets_list:
            if dataset_name not in self.dataset_info_map:
                print(f"⚠️  警告: 未知数据集 '{dataset_name}', 跳过")
                continue
            
            # 创建训练集和测试集
            train_dataset = GenericHARDataset(
                root_dir=root_dir,
                dataset_name=dataset_name,
                split='train',
                transform=transform
            )
            
            test_dataset = GenericHARDataset(
                root_dir=root_dir,
                dataset_name=dataset_name,
                split='test',
                transform=None  # 测试集不做数据增强
            )
            
            # 创建数据加载器
            dataloaders[dataset_name] = {
                'train': DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory
                ),
                'test': DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
            }
        
        return dataloaders
    
    def get_input_size(self, dataset_name: str, batch_size: int = 1) -> tuple:
        """
        获取输入尺寸
        
        返回:
            (batch_size, channels, time_steps)
        """
        info = self.get_dataset_info(dataset_name)
        if not info:
            raise ValueError(f"未找到数据集 '{dataset_name}' 的信息")
        
        return (batch_size, info['channels'], info['time_steps'])
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        return self.dataset_info_map.get(dataset_name, {})
    
    def create_calibration_loader(
        self, 
        dataloader: DataLoader, 
        num_batches: int = 5
    ) -> DataLoader:
        """
        创建校准数据加载器
        
        参数:
            dataloader: 原始数据加载器
            num_batches: 校准批次数
        
        返回:
            校准数据加载器
        """
        batch_size = dataloader.batch_size
        total_samples = num_batches * batch_size
        dataset = dataloader.dataset
        subset_indices = list(range(min(total_samples, len(dataset))))
        
        calibration_dataset = Subset(dataset, subset_indices)
        
        return DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
    
    def list_available_datasets(self) -> List[str]:
        """列出所有可用的数据集"""
        return list(self.dataset_info_map.keys())
    
    def print_dataset_info(self, dataset_name: str):
        """打印数据集详细信息"""
        info = self.get_dataset_info(dataset_name)
        if not info:
            print(f"❌ 未找到数据集 '{dataset_name}' 的信息")
            return
        
        print(f"\n{'='*60}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*60}")
        print(f"描述: {info['description']}")
        print(f"输入形状: (batch_size, {info['channels']}, {info['time_steps']})")
        print(f"通道数: {info['channels']}")
        print(f"时间步长: {info['time_steps']}")
        print(f"类别数: {info['num_classes']}")
        print(f"{'='*60}\n")


# 兼容旧代码的便捷函数
def get_har_dataloaders(root_dir: str, batch_size: int = 64, 
                        datasets: Optional[List[str]] = None,
                        num_workers: int = 0, pin_memory: bool = False):
    """
    兼容旧代码的便捷函数
    
    参数:
        root_dir: 数据集根目录
        batch_size: 批次大小
        datasets: 数据集名称列表
        num_workers: 工作进程数
        pin_memory: 是否固定内存
    
    返回:
        数据加载器字典
    """
    loader = HARDataLoader()
    return loader.get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        datasets_list=datasets,
        num_workers=num_workers,
        pin_memory=pin_memory
    )