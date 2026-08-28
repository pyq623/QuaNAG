# image_loader.py
"""
2D图像数据加载器
"""
from typing import Dict, Any, Optional, List
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets

from .base_loader import BaseDataLoader, DataLoaderFactory


# 2D图像数据集信息配置
IMAGE_DATASET_INFO = {
    # ==================== CIFAR系列 ====================
    'cifar10': {
        'task_type': 'image',  # ✅ 添加
        'channels': 3,           # RGB 彩色图像
        'height': 32,            # 图像高度
        'width': 32,             # 图像宽度
        'num_classes': 10,
        'description': 'CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 50,000 training and 10,000 test images',
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'input_shape': (3, 32, 32),  # (C, H, W)
        'normalization': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        }
    },
    
    'cifar100': {
        'task_type': 'image',  # ✅ 添加
        'channels': 3,
        'height': 32,
        'width': 32,
        'num_classes': 100,
        'description': 'CIFAR-100 dataset contains 60,000 32x32 color images in 100 fine-grained classes grouped into 20 superclasses, with 50,000 training and 10,000 test images',
        'superclasses': 20,      # 20个超类
        'input_shape': (3, 32, 32),
        'normalization': {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    },
    
    # ==================== MNIST系列 ====================
    'mnist': {
        'task_type': 'image',  # ✅ 添加
        'channels': 1,           # 灰度图像
        'height': 28,
        'width': 28,
        'num_classes': 10,
        'description': 'MNIST dataset contains 70,000 28x28 grayscale images of handwritten digits (0-9), with 60,000 training and 10,000 test images',
        'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'input_shape': (1, 28, 28),
        'normalization': {
            'mean': [0.1307],
            'std': [0.3081]
        }
    },
    
    # ==================== ImageNet系列 ====================
    'imagenet': {
        'task_type': 'image',  # ✅ 添加
        'channels': 3,
        'height': 224,           # 标准输入尺寸
        'width': 224,
        'num_classes': 1000,
        'description': 'ImageNet dataset contains over 1.2 million training images and 50,000 validation images in 1000 classes',
        'input_shape': (3, 224, 224),
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    },
    
    'tiny-imagenet': {
        'task_type': 'image',  # ✅ 添加
        'channels': 3,
        'height': 64,            # Tiny-ImageNet尺寸
        'width': 64,
        'num_classes': 200,
        'description': 'Tiny-ImageNet dataset is a subset of ImageNet with 200 classes, containing 100,000 64x64 training images and 10,000 validation images',
        'input_shape': (3, 64, 64),
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
}


class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset Wrapper"""
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化CIFAR-10数据集
        
        参数:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            transform (callable, optional): 数据变换
        """
        self.root_dir = root_dir
        self.split = split
        
        # 使用torchvision加载CIFAR-10
        is_train = (split == 'train')
        self.dataset = datasets.CIFAR10(
            root=root_dir,
            train=is_train,
            download=True,  # 如果数据不存在会自动下载
            transform=None  # 我们稍后手动应用transform
        )
        
        self.transform = transform
        
        # CIFAR-10的类别
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 创建label字典
        self.label_dict = {i: class_name for i, class_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 获取PIL图像和标签
        image, label = self.dataset[idx]
        
        # 将PIL图像转换为tensor [C, H, W]
        # CIFAR-10图像是32x32的RGB图像
        image = transforms.ToTensor()(image)  # [3, 32, 32]
        
        # 应用自定义transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CIFAR100Dataset(Dataset):
    """CIFAR-100 Dataset Wrapper"""
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化CIFAR-100数据集
        
        参数:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            transform (callable, optional): 数据变换
        """
        self.root_dir = root_dir
        self.split = split
        
        is_train = (split == 'train')
        self.dataset = datasets.CIFAR100(
            root=root_dir,
            train=is_train,
            download=True,
            transform=None
        )
        
        self.transform = transform
        self.classes = self.dataset.classes
        self.label_dict = {i: class_name for i, class_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = transforms.ToTensor()(image)  # [3, 32, 32]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MNISTDataset(Dataset):
    """MNIST Dataset Wrapper"""
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化MNIST数据集
        
        参数:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'test'
            transform (callable, optional): 数据变换
        """
        self.root_dir = root_dir
        self.split = split
        
        is_train = (split == 'train')
        self.dataset = datasets.MNIST(
            root=root_dir,
            train=is_train,
            download=True,
            transform=None
        )
        
        self.transform = transform
        self.classes = [str(i) for i in range(10)]
        self.label_dict = {i: str(i) for i in range(10)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = transforms.ToTensor()(image)  # [1, 28, 28]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImageNetSubset(Dataset):
    """ImageNet Subset (如Tiny-ImageNet)"""
    def __init__(self, root_dir, split='train', transform=None, image_size=64):
        """
        初始化ImageNet子集
        
        参数:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'val'
            transform (callable, optional): 数据变换
            image_size (int): 图像尺寸（默认64用于Tiny-ImageNet）
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # 这里假设使用ImageFolder结构
        split_dir = 'train' if split == 'train' else 'val'
        self.dataset = datasets.ImageFolder(
            root=os.path.join(root_dir, split_dir),
            transform=None
        )
        
        self.classes = self.dataset.classes
        self.label_dict = {i: class_name for i, class_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # 调整图像尺寸
        resize_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        image = resize_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


@DataLoaderFactory.register('image')
class ImageDataLoader(BaseDataLoader):
    """图像数据加载器实现"""
    
    def __init__(self, task_type: str = 'image'):
        super().__init__(task_type)
        self.dataset_info_map = IMAGE_DATASET_INFO
    
    def get_dataloaders(
        self, 
        root_dir: str, 
        batch_size: int = 64, 
        datasets_list: Optional[List[str]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, DataLoader]]:
        """
        获取图像数据加载器
        
        参数:
            root_dir: 数据集根目录
            batch_size: 批次大小
            datasets_list: 数据集名称列表
            num_workers: 数据加载工作进程数
            pin_memory: 是否固定内存
        
        返回:
            {dataset_name: {'train': DataLoader, 'test': DataLoader}}
        """
        # 默认加载CIFAR-10
        if datasets_list is None:
            datasets_list = ['cifar10']
        
        dataloaders = {}
        
        for dataset_name in datasets_list:
            dataset_name_lower = dataset_name.lower()
            
            if dataset_name_lower not in self.dataset_info_map:
                print(f"⚠️  警告: 未知数据集 '{dataset_name}', 跳过")
                continue
            
            # 获取归一化参数
            norm_params = self.dataset_info_map[dataset_name_lower].get('normalization', {})
            
            if dataset_name_lower in ['cifar10', 'cifar100']:
                # CIFAR数据增强
                normalize = transforms.Normalize(
                    mean=norm_params.get('mean', [0.5, 0.5, 0.5]),
                    std=norm_params.get('std', [0.5, 0.5, 0.5])
                )
                
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    normalize
                ])
                
                test_transform = transforms.Compose([
                    normalize
                ])
                
                # 创建数据集
                if dataset_name_lower == 'cifar10':
                    train_dataset = CIFAR10Dataset(root_dir, 'train', train_transform)
                    test_dataset = CIFAR10Dataset(root_dir, 'test', test_transform)
                else:
                    train_dataset = CIFAR100Dataset(root_dir, 'train', train_transform)
                    test_dataset = CIFAR100Dataset(root_dir, 'test', test_transform)
            
            elif dataset_name_lower == 'mnist':
                # MNIST数据增强
                normalize = transforms.Normalize(
                    mean=norm_params.get('mean', [0.5]),
                    std=norm_params.get('std', [0.5])
                )
                
                train_transform = transforms.Compose([
                    transforms.RandomRotation(10),
                    normalize
                ])
                
                test_transform = transforms.Compose([
                    normalize
                ])
                
                train_dataset = MNISTDataset(root_dir, 'train', train_transform)
                test_dataset = MNISTDataset(root_dir, 'test', test_transform)
            
            else:
                raise ValueError(f"暂不支持数据集: {dataset_name}")
            
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
            (batch_size, channels, height, width)
        """
        info = self.get_dataset_info(dataset_name)
        if not info:
            # 默认返回CIFAR-10尺寸
            return (batch_size, 3, 32, 32)
        
        return (batch_size, info['channels'], info['height'], info['width'])
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        dataset_name_lower = dataset_name.lower()
        return self.dataset_info_map.get(dataset_name_lower, {})
    
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
        print(f"数据集: {dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"描述: {info['description']}")
        print(f"输入形状: {info['input_shape']}")
        print(f"通道数: {info['channels']}")
        print(f"图像尺寸: {info['height']}x{info['width']}")
        print(f"类别数: {info['num_classes']}")
        
        if 'classes' in info:
            classes_str = ', '.join(info['classes'][:5])
            if len(info['classes']) > 5:
                classes_str += '...'
            print(f"类别: {classes_str}")
        
        if 'normalization' in info:
            print(f"归一化参数:")
            print(f"  Mean: {info['normalization']['mean']}")
            print(f"  Std: {info['normalization']['std']}")
        
        print(f"{'='*60}\n")


# 兼容旧代码的便捷函数
def get_image_dataloaders(root_dir: str, batch_size: int = 64, 
                           datasets_list: Optional[List[str]] = None,
                           num_workers: int = 4, pin_memory: bool = True):
    """
    兼容旧代码的便捷函数
    
    参数:
        root_dir: 数据集根目录
        batch_size: 批次大小
        datasets_list: 数据集名称列表
        num_workers: 工作进程数
        pin_memory: 是否固定内存
    
    返回:
        数据加载器字典
    """
    loader = ImageDataLoader()
    return loader.get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        datasets_list=datasets_list,
        num_workers=num_workers,
        pin_memory=pin_memory
    )