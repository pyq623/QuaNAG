# /root/unified_nas/models/candidate_models.py
"""
统一的候选模型管理
支持1D时序数据(HAR)和2D图像数据(Image)
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from .base_model import TinyMLModel
from data import get_data_loader  # 使用统一的数据加载器

# 设置随机数种子
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


@dataclass
class CandidateModel:
    """
    统一的候选神经网络架构及其评估指标
    
    通过task_type参数支持HAR和Image任务
    
    属性:
        config: 模型架构配置字典
        task_type: 任务类型 ('har' 或 'image')
        candidate_id: 候选模型唯一标识符 (字符串)
        accuracy: 验证准确率 (0-1)
        latency: 推理延迟 (ms)
        generation: 进化算法中的生成代数
        parent_ids: 父代ID列表 (用于遗传算法)
        metadata: 其他元数据
    """
    config: Dict[str, Any]
    task_type: str = 'har'  # ✅ 新增：任务类型
    candidate_id: Optional[str] = None
    accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    generation: Optional[int] = None
    parent_ids: Optional[List[int]] = None
    metrics: Optional[Dict[str, Any]] = None
    latency: Optional[float] = None
    peak_memory: Optional[float] = None
    estimate_total_size: Optional[float] = None
    comparison_metrics: Optional[Dict[str, float]] = None
    use_quantized_metrics: Optional[bool] = None

    def __post_init__(self):
        """数据验证和默认值设置"""
        self.parent_ids = self.parent_ids or []
        self.metrics = self.metrics or {}
        self.val_accuracy = self.val_accuracy or {}
        self.comparison_metrics = self.comparison_metrics or {}
        self.use_quantized_metrics = self.use_quantized_metrics or False

        # 如果没有提供 candidate_id，生成一个默认的
        if self.candidate_id is None:
            import uuid
            self.candidate_id = f"candidate_{uuid.uuid4().hex[:8]}"

    def build_model(self) -> nn.Module:
        """
        将配置转换为PyTorch模型
        
        返回:
            根据task_type构建的模型
        """
        # ✅ 传入task_type参数
        model = TinyMLModel(self.config, task_type=self.task_type)

        # 确保模型有 output_dim 属性
        if not hasattr(model, 'output_dim'):
            model.output_dim = self._calculate_output_dim()

        return model

    def _calculate_output_dim(self) -> int:
        """根据配置计算最终输出维度"""
        if 'stages' not in self.config:
            return 64  # 默认值
        
        # 取最后一个 stage 的通道数作为输出维度
        last_stage = self.config['stages'][-1]
        return int(last_stage['channels'])
    
    def measure_peak_memory(self, device='cuda', dataset_names=None, 
                           data_root=None) -> float:
        """
        测量模型运行时的峰值内存（单位：MB）
        
        参数:
            device: 设备 ('cuda' 或 'cpu')
            dataset_names: 数据集名称列表
            data_root: 数据根目录（如果为None，使用默认路径）
        """
        # ✅ 使用统一的数据加载器
        loader = get_data_loader(self.task_type)
        
        # 设置默认数据路径
        if data_root is None:
            if self.task_type == 'har':
                data_root = '/root/har_train/data/UniMTS_data'
            else:
                data_root = '/root/image_train/data'
        
        # 设置默认数据集
        if dataset_names is None:
            if self.task_type == 'har':
                dataset_names = ['MMAct']
            else:
                dataset_names = ['cifar10']
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        elif not isinstance(dataset_names, list):
            raise ValueError(f"Invalid dataset_names type: {type(dataset_names)}")

        # 加载数据
        dataloaders = loader.get_dataloaders(
            root_dir=data_root,
            batch_size=64,
            datasets_list=dataset_names
        )

        model = self.build_model().to(device)
        model.eval()

        total_peak_memory = 0
        total_samples = 0
        max_memory = 0

        for dataset_name in dataset_names:
            print(f"测量数据集 {dataset_name} 的峰值内存...")
            dataloader = dataloaders[dataset_name]['train']
            dataset_peak_memory = 0

            for i, (inputs, _) in enumerate(dataloader):
                if i >= 100:  # 只测量前 100 条数据
                    break

                inputs = inputs.to(device)

                if device == 'cuda':
                    # 清空显存缓存并重置统计
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                    # 前向传播
                    with torch.no_grad():
                        _ = model(inputs)

                    # 获取峰值内存
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                elif device == 'cpu':
                    import tracemalloc
                    tracemalloc.start()

                    # 前向传播
                    with torch.no_grad():
                        _ = model(inputs)

                    # 获取内存使用
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    peak_memory = peak / (1024 ** 2)
                else:
                    raise ValueError(f"Unsupported device: {device}")

                dataset_peak_memory += peak_memory
                total_samples += 1
                if max_memory < peak_memory:    
                    max_memory = peak_memory

            avg_dataset_peak_memory = dataset_peak_memory / min(100, len(dataloader))
            print(f"数据集 {dataset_name} 的平均峰值内存: {avg_dataset_peak_memory:.2f} MB")
            print(f"数据集 {dataset_name} 的最大峰值内存: {max_memory:.2f} MB")
            total_peak_memory += avg_dataset_peak_memory

        self.peak_memory = max_memory
        return self.peak_memory

    @classmethod
    def load(cls, file_path: str, task_type: str = 'har') -> "CandidateModel":
        """
        从JSON文件加载候选模型
        
        参数:
            file_path: JSON文件路径
            task_type: 任务类型 ('har' 或 'image')
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            config=data["config"],
            task_type=task_type,  # ✅ 传入task_type
            candidate_id=data.get("candidate_id"),
            accuracy=data["metrics"].get("accuracy"),
            latency=data["metrics"].get("latency"),
            generation=data.get("generation"),
            parent_ids=data.get("parent_ids", []),
            metrics=data.get("metrics", {})
        )
    
    def get_details(self) -> Dict[str, Any]:
        """
        返回CandidateModel的详细信息，包括config和所有主要属性。
        """
        return {
            "config": self.config,
            "task_type": self.task_type,  # ✅ 包含task_type
            "candidate_id": self.candidate_id,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "peak_memory": self.peak_memory,
            "val_accuracy": self.val_accuracy,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metrics": self.metrics,
        }