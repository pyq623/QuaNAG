# /root/unified_nas/mcts/config_generator.py
"""
统一的配置生成器
支持HAR和Image任务的模型配置生成

核心特性:
- 自动适配task_type (har/image)
- 内存约束检查
- 智能种子生成 (按量化模式分配)
- 多样化的配置生成
"""

import json
import random
from typing import Dict, Any, List
import sys

from models import CandidateModel


class ConfigGenerator:
    """
    模型配置生成器
    
    支持HAR和Image任务，自动根据dataset_info中的task_type进行适配
    """
    
    def __init__(self, search_space: Dict[str, Any], dataset_info: Dict[str, Any], 
                 max_memory: float = 20.0, task_type: str = 'har'):
        """
        初始化配置生成器
        
        参数:
            search_space: 搜索空间定义
            dataset_info: 数据集信息 (需包含task_type或根据字段自动推断)
            max_memory: 最大内存限制 (MB)
            task_type: 任务类型 ('har' 或 'image')
        """
        self.search_space = search_space
        self.dataset_info = dataset_info
        self.max_memory = max_memory
        
        # ✅ 自动推断或使用提供的task_type
        self.task_type = dataset_info.get('task_type', task_type)
        
        # ✅ 验证dataset_info包含必需字段
        self._validate_dataset_info()
        
    def _validate_dataset_info(self):
        """验证dataset_info包含task_type所需的字段"""
        required_fields = ['channels', 'num_classes']
        
        if self.task_type == 'har':
            required_fields.append('time_steps')
        elif self.task_type == 'image':
            required_fields.extend(['height', 'width'])
        else:
            raise ValueError(f"不支持的task_type: {self.task_type}")
        
        for field in required_fields:
            if field not in self.dataset_info:
                raise ValueError(f"dataset_info缺少必需字段: {field}")
        
    def generate_random_config(self, quant_mode: str = "none") -> Dict[str, Any]:
        """
        生成随机模型配置，确保满足内存约束
        
        参数:
            quant_mode: 量化模式 ('none', 'static', 'qat', 'qaft')
        
        返回:
            配置字典
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            config = self._try_generate_random_config(quant_mode)
            if self._check_memory_constraint(config):
                return config
        
        # 如果多次尝试都失败，返回最小配置
        return self._generate_minimal_config(quant_mode)
    
    def _try_generate_random_config(self, quant_mode: str = "none") -> Dict[str, Any]:
        """
        生成随机模型配置
        
        参数:
            quant_mode: 量化模式
        
        返回:
            配置字典
        """
        # 限制阶段数量和block数量来满足内存约束
        max_stages = min(4, max(self.search_space["stages"]))
        max_blocks_per_stage = min(2, max(self.search_space["blocks_per_stage"]))

        num_stages = random.choice([s for s in self.search_space["stages"] 
                                   if s <= max_stages])
        
        stages = []
        in_channels = self.dataset_info.get("channels", 6 if self.task_type == 'har' else 3)
        classes = self.dataset_info.get("num_classes", 10)
        current_channels = in_channels
        
        for i in range(num_stages):
            stage_channels = random.choice(self.search_space["channels"])
            num_blocks = random.choice([b for b in self.search_space["blocks_per_stage"] 
                                       if b <= max_blocks_per_stage])
            
            blocks = []
            for j in range(num_blocks):
                is_first_block = (j == 0)
                block = self._generate_block(current_channels, stage_channels, is_first_block)
                blocks.append(block)
                
                # 更新当前通道数
                if block["type"] == "SeDpConv":
                    current_channels = stage_channels
                else:
                    current_channels = stage_channels
            
            stage_config = {
                "blocks": blocks,
                "channels": stage_channels
            }
            stages.append(stage_config)
        
        return {
            "input_channels": in_channels,
            "num_classes": classes,
            "quant_mode": quant_mode,
            "stages": stages
        }
    
    def _generate_block(self, in_channels: int, stage_channels: int, 
                       is_first_block: bool) -> Dict[str, Any]:
        """
        生成单个block配置
        
        参数:
            in_channels: 输入通道数
            stage_channels: 阶段通道数
            is_first_block: 是否为第一个block
        
        返回:
            block配置字典
        """
        block_type = random.choice(self.search_space["conv_types"])
        
        # 处理SeDpConv的特殊约束
        if block_type == "SeDpConv":
            # SeDpConv要求输入输出通道相等
            stage_channels = in_channels
            # 如果是第一个block，通道数必须等于input_channels
            if is_first_block:
                stage_channels = self.dataset_info.get("channels", 
                                                      6 if self.task_type == 'har' else 3)
        
        # 处理MBConv和DWSepConv的关系
        if block_type == "MBConv":
            expansion = random.choice([e for e in self.search_space["expansions"] 
                                     if 1 < e <= 3])
        elif block_type == "DWSepConv":
            expansion = 1
        else:
            expansion = random.choice([e for e in self.search_space["expansions"] 
                                     if e <= 2])
        
        # 处理SE模块
        has_se = random.choice(self.search_space["has_se"])
        if has_se:
            se_ratio = random.choice([r for r in self.search_space["se_ratios"] if r > 0])
        else:
            se_ratio = 0
        
        # 对于SeDpConv和SeSepConv，强制启用SE
        if block_type in ["SeDpConv", "SeSepConv"]:
            has_se = True
            se_ratio = random.choice([r for r in self.search_space["se_ratios"] if r > 0])
        
        # 处理skip connection规则
        if block_type in ["DpConv", "SeDpConv", "SeSepConv"]:
            # 这些类型不支持skip connection
            skip_connection = False
        else:
            # DWSepConv、MBConv支持skip connection
            skip_connection = random.choice(self.search_space["skip_connection"])

        return {
            "type": block_type,
            "kernel_size": random.choice(self.search_space["kernel_sizes"]),
            "stride": random.choice(self.search_space["strides"]) if is_first_block else 1,
            "expansion": expansion,
            "has_se": has_se,
            "se_ratio": se_ratio,
            "skip_connection": skip_connection,
            "activation": random.choice(self.search_space["activations"])
        }
    
    def _check_memory_constraint(self, config: Dict[str, Any]) -> bool:
        """
        检查配置的内存使用情况
        
        参数:
            config: 模型配置
        
        返回:
            是否满足内存约束
        """
        try:
            # ✅ 传递task_type到CandidateModel
            candidate = CandidateModel(config=config, task_type=self.task_type)
            model = candidate.build_model()
            
            # ✅ 根据task_type构建不同的输入尺寸
            if self.task_type == 'har':
                # HAR: (batch, channels, time_steps)
                input_size = (64, 
                            self.dataset_info['channels'], 
                            self.dataset_info['time_steps'])
            else:  # image
                # Image: (batch, channels, height, width)
                input_size = (1, 
                            self.dataset_info['channels'], 
                            self.dataset_info['height'], 
                            self.dataset_info['width'])
            
            # 导入utils模块进行内存计算
            from utils import calculate_memory_usage
            memory_info = calculate_memory_usage(model, input_size=input_size, device='cpu')
            memory_usage = memory_info["total_memory_MB"]

            # 根据量化模式调整内存使用量
            quant_mode = config.get("quant_mode", "none")
            if quant_mode in ["static", "qat", "qaft"]:
                # 量化模型通常可以压缩到原来的1/4左右
                compressed_memory = memory_usage / 4.0
                memory_usage = compressed_memory

            return memory_usage <= self.max_memory
            
        except Exception as e:
            # 如果检查失败，返回False
            return False
        
    def _generate_minimal_config(self, quant_mode: str) -> Dict[str, Any]:
        """
        生成最小有效配置（保底方案）
        
        参数:
            quant_mode: 量化模式
        
        返回:
            最小配置字典
        """
        in_channels = self.dataset_info.get("channels", 
                                           6 if self.task_type == 'har' else 3)
        classes = self.dataset_info.get("num_classes", 10)
        
        return {
            "input_channels": in_channels,
            "num_classes": classes,
            "quant_mode": quant_mode,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "SeDpConv",
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": True,
                            "se_ratio": 0.25,
                            "skip_connection": False,
                            "activation": "ReLU6"
                        }
                    ],
                    "channels": in_channels
                }
            ]
        }
    
    def _generate_random_stage(self, in_channels: int) -> Dict[str, Any]:
        """
        生成随机阶段配置
        
        参数:
            in_channels: 输入通道数
        
        返回:
            阶段配置字典
        """
        stage_channels = random.choice(self.search_space["channels"])
        max_blocks = min(2, max(self.search_space["blocks_per_stage"]))
        num_blocks = random.choice([b for b in self.search_space["blocks_per_stage"] 
                                   if b <= max_blocks])
        
        blocks = []
        current_channels = in_channels
        for i in range(num_blocks):
            block = self._generate_block(current_channels, stage_channels, i == 0)
            blocks.append(block)
            if block["type"] == "SeDpConv":
                current_channels = stage_channels
            else:
                current_channels = stage_channels
        
        return {
            "blocks": blocks,
            "channels": stage_channels
        }


class ScatteringGenerator:
    """
    散射生成器 - 生成多样化的初始配置
    
    使用智能的种子分配策略：
    - 前16个种子按量化模式分配 (4种量化 × 4种卷积类型)
    - 后续种子随机生成
    """
    
    def __init__(self, config_generator: ConfigGenerator):
        """
        初始化散射生成器
        
        参数:
            config_generator: 配置生成器实例
        """
        self.config_generator = config_generator
        
        # ✅ 定义5种卷积类型
        self.conv_types = ["DpConv", "DWSepConv", "SeSepConv", "MBConv", "SeDpConv"]
        
        # ✅ 定义4种量化模式
        self.quant_modes = ["static", "none", "qat", "qaft"]
    
    def generate_scattered_seeds(self, num_seeds: int) -> List[Dict[str, Any]]:
        """
        生成散射的初始种子配置 - 按量化模式分配
        
        策略：
        - 种子 0-3:   static 量化 (4个不同conv类型)
        - 种子 4-7:   none 量化   (4个不同conv类型)
        - 种子 8-11:  qat 量化    (4个不同conv类型)
        - 种子 12-15: qaft 量化   (4个不同conv类型)
        - 种子 16+:   随机生成
        
        每组尝试覆盖不同的卷积类型
        
        参数:
            num_seeds: 需要生成的种子数量
        
        返回:
            配置列表
        """
        seeds = []

        print(f"\n📋 生成 {num_seeds} 个按量化模式分配的种子:")
        sys.stdout.flush()
        
        # ✅ 前16个种子：4种量化模式 × 4个种子/模式
        for i in range(min(16, num_seeds)):
            # 计算当前种子的量化模式和卷积类型
            quant_mode_idx = i // 4  # 0-3 → 每4个种子一个量化模式
            conv_type_idx = i % 4    # 0-3 → 循环选择前4种卷积类型
            
            quant_mode = self.quant_modes[quant_mode_idx]
            preferred_conv_type = self.conv_types[conv_type_idx]
            
            print(f"  生成种子 {i}: quant_mode={quant_mode}, "
                  f"preferred_type={preferred_conv_type}")
            sys.stdout.flush()
            
            # 生成配置
            config = self._generate_with_preferred_type(preferred_conv_type, quant_mode)
            seeds.append(config)
        
        # ✅ 如果需要超过16个种子，剩余的随机生成
        if num_seeds > 16:
            print(f"\n  生成额外 {num_seeds - 16} 个随机种子 "
                  f"(种子 16-{num_seeds-1}):")
            sys.stdout.flush()
            
            for i in range(16, num_seeds):
                # 随机选择量化模式
                quant_mode = random.choice(self.quant_modes)
                config = self.config_generator.generate_random_config(quant_mode)
                seeds.append(config)
                
                seed_type = config["stages"][0]["blocks"][0]["type"]
                print(f"    种子 {i}: quant_mode={quant_mode}, type={seed_type}")
                sys.stdout.flush()
        
        return seeds
        
    def _generate_with_preferred_type(self, preferred_type: str, 
                                      quant_mode: str) -> Dict[str, Any]:
        """
        生成偏好某种类型的配置（但不强制）
        
        参数:
            preferred_type: 偏好的卷积类型
            quant_mode: 量化模式
        
        返回:
            配置字典
        """
        max_attempts = 8  # ✅ 增加尝试次数（量化模式放宽了内存限制）
        
        for attempt in range(max_attempts):
            # 生成随机配置
            config = self.config_generator._try_generate_random_config(quant_mode)
            
            # 检查第一个block是否是目标类型
            if config["stages"][0]["blocks"][0]["type"] == preferred_type:
                # 检查内存
                if self.config_generator._check_memory_constraint(config):
                    print(f"    ✅ 尝试 {attempt+1}/{max_attempts}: "
                          f"成功生成 {preferred_type}")
                    sys.stdout.flush()
                    return config
        
        # ✅ 如果多次尝试都没成功，使用完全随机（保持量化模式）
        print(f"    ⚠️  未能生成 {preferred_type}，使用随机配置")
        sys.stdout.flush()
        fallback_config = self.config_generator.generate_random_config(quant_mode)
        return fallback_config