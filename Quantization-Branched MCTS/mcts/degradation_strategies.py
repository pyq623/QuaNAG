"""
配置降级策略模块
用于在内存超标时自动降级模型配置

支持HAR和Image任务的统一降级策略
"""
import json
from typing import Dict, Any, Optional, Callable
import random
import sys


class ConfigDegradationManager:
    """
    配置降级策略管理器 - 统一版
    
    当LLM生成的配置不满足约束时，智能降级以满足内存限制
    支持HAR和Image任务
    """
    
    def __init__(self, dataset_info: Dict[str, Any], max_memory: float,
                 check_memory_fn: Callable, is_duplicate_fn: Callable,
                 task_type: str = 'har'):
        """
        初始化降级管理器
        
        Args:
            dataset_info: 数据集信息（需包含task_type或channels/time_steps/height/width）
            max_memory: 最大内存限制(MB)
            check_memory_fn: 内存检查函数
            is_duplicate_fn: 重复检查函数
            task_type: 任务类型 ('har' 或 'image')
        """
        self.dataset_info = dataset_info
        self.max_memory = max_memory
        self.check_memory_fn = check_memory_fn
        self.is_duplicate_fn = is_duplicate_fn
        
        # ✅ 自动推断task_type
        self.task_type = dataset_info.get('task_type', task_type)
        
        # 合法的通道数选项
        self.valid_channels = [8, 16, 24, 32]

        # 降级策略列表
        self.strategies = [
            self.apply_stage_reduction,
            self.apply_conv_type_optimization,
            self.apply_channel_reduction,
            self.apply_block_reduction
        ]

        print(f"🛠️ 降级管理器初始化: task_type={self.task_type}, "
              f"最大内存={max_memory:.2f}MB")

    def generate_degraded_config(self, base_config: Dict[str, Any], 
                                 direction: str, current_memory: float,
                                 dup_modification: str,
                                 dialogue_record: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成降级配置
        
        Args:
            base_config: 基础配置
            direction: 量化方向
            current_memory: 当前内存使用(MB)
            dup_modification: 强制修改模式（用于去重，即使内存合规也修改）
            dialogue_record: 可选的对话记录（用于记录降级过程）
        
        Returns:
            降级后的配置
        """
        print("🛠️ 使用智能降级生成安全配置")
        print(f"   - 任务类型: {self.task_type}")
        print(f"   - 基础配置: {len(base_config.get('stages', []))} stages")
        sys.stdout.flush() 
        print(f"   - 当前内存: {current_memory:.2f}MB")
        sys.stdout.flush() 
        print(f"   - 目标内存: {self.max_memory:.2f}MB")
        sys.stdout.flush() 
        
        # ✅ 记录降级模式
        if dup_modification:
            print(f"   - 模式: 🔄 修改去重")
        else:
            print(f"   - 模式: 📉 内存降级")
        sys.stdout.flush()
        
        # ✅ 记录降级触发原因（如果有对话记录）
        if dialogue_record:
            degradation_reason = {
                "trigger": "memory_violation" if current_memory > self.max_memory else "duplicate_config",
                "current_memory_MB": current_memory,
                "max_memory_MB": self.max_memory,
                "modification_mode": dup_modification,
                "task_type": self.task_type
            }
            feedback_msg = f"""
            --- DEGRADATION TRIGGERED (MCTS Auto-Correction) ---
            **Reason:** {degradation_reason['trigger'].upper()}
            - Task Type: {self.task_type}
            - Current Memory: {current_memory:.2f} MB
            - Memory Limit: {self.max_memory:.2f} MB
            - Mode: {'Duplicate Removal' if dup_modification else 'Memory Reduction'}

            **ACTION:** Applying intelligent degradation strategies...
            """
            dialogue_record["messages"].append({
                "role": "user",
                "content": feedback_msg
            })
        
        # 创建安全配置
        safe_config = base_config.copy() if base_config else {}
        safe_config["quant_mode"] = direction
        
        # 如果没有基础配置，生成最小配置
        if not safe_config.get("stages"):
            safe_config = self._create_minimal_config_safe(direction)
            if not self.is_duplicate_fn(safe_config):
                # ✅ 记录降级结果
                if dialogue_record:
                    dialogue_record["messages"].append({
                        "role": "assistant",
                        "content": f"```json\n{json.dumps(safe_config, indent=2)}\n```\n**Degradation: Minimal safe config generated.**"
                    })
                return safe_config
        
        # 依次尝试降级策略
        for strategy in self.strategies:
            temp_config = strategy(safe_config.copy(), current_memory, dup_modification)
            if temp_config and not self.is_duplicate_fn(temp_config):
                memory_ok, memory_usage, _ = self.check_memory_fn(temp_config)
                if memory_ok:
                    print(f"✅ 降级策略成功: 内存={memory_usage:.2f}MB")
                    sys.stdout.flush() 
                    # ✅ 记录成功的降级配置
                    if dialogue_record:
                        dialogue_record["messages"].append({
                            "role": "assistant",
                            "content": f"```json\n{json.dumps(temp_config, indent=2)}\n```\n**Degradation SUCCESS: Memory={memory_usage:.2f}MB**"
                        })
                    return temp_config
        
        # 最终手段：绝对最小配置
        minimal_config = self._create_minimal_config_with_retry_fixed(direction)
        
        # ✅ 记录最终降级
        if dialogue_record:
            dialogue_record["messages"].append({
                "role": "assistant",
                "content": f"```json\n{json.dumps(minimal_config, indent=2)}\n```\n**Degradation: Emergency minimal config.**"
            })
        
        return minimal_config
    
    def _create_minimal_config_safe(self, direction: str) -> Dict[str, Any]:
        """
        创建安全的最小配置
        
        ✅ 使用DpConv而非SeDpConv，避免严格的通道约束
        """
        print("🚨 使用安全最小配置 (DpConv)")
        
        return {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "DpConv",  # ✅ 使用DpConv（低内存且无严格约束）
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "se_ratio": 0,
                            "activation": "ReLU6",
                            "skip_connection": False
                        }
                    ],
                    "channels": 8  # ✅ 使用最小通道数
                }
            ]
        }

    def _create_minimal_config(self, direction: str) -> Dict[str, Any]:
        """创建最小配置（使用SeDpConv）"""
        return {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "SeDpConv",  # 最小内存的卷积类型
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": True,
                            "se_ratio": 0.25,
                            "activation": "ReLU",
                            "skip_connection": False
                        }
                    ],
                    "channels": self.dataset_info["channels"]
                }
            ]
        }
    
    def _create_minimal_config_with_retry_fixed(self, direction: str) -> Dict[str, Any]:
        """
        创建最小配置并处理重复 - 完全修复版
        
        ✅ 使用随机扰动生成不同配置
        """
        print("🚨 使用最小配置（带智能重试）")
        
        # 尝试多种不同的最小配置
        attempt = 0
        max_attempts = 20
        
        while attempt < max_attempts:
            # ✅ 生成多样化的最小配置
            if attempt == 0:
                config = self._create_minimal_config_safe(direction)
            elif attempt == 1:
                config = self._create_minimal_dwsepconv_config(direction)
            elif attempt < 10:
                config = self._create_minimal_config_with_variation(direction, attempt)
            else:
                config = self._create_random_minimal_config(direction)
            
            # 检查是否重复
            if not self.is_duplicate_fn(config):
                # 验证配置有效性
                try:
                    memory_ok, memory_usage, msg = self.check_memory_fn(config)
                    if memory_ok:
                        print(f"✅ 生成有效配置 (尝试 {attempt+1}/{max_attempts}): "
                              f"{memory_usage:.2f}MB")
                        return config
                    else:
                        print(f"⚠️  配置内存超标 (尝试 {attempt+1}): "
                              f"{memory_usage:.2f}MB")
                except Exception as e:
                    print(f"⚠️  配置验证失败 (尝试 {attempt+1}): {e}")
            else:
                print(f"🔁 配置重复 (尝试 {attempt+1}/{max_attempts})")
            
            attempt += 1
        
        # 最终回退
        print("⚠️  所有尝试失败，返回基础配置")
        return self._create_minimal_config_safe(direction)
    
    def _create_minimal_dwsepconv_config(self, direction: str) -> Dict[str, Any]:
        """创建使用DWSepConv的最小配置"""
        return {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": "DWSepConv",
                            "kernel_size": 3,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "se_ratio": 0,
                            "activation": "ReLU6",
                            "skip_connection": False
                        }
                    ],
                    "channels": 8
                }
            ]
        }
    
    def _create_minimal_config_with_variation(self, direction: str, 
                                              seed: int) -> Dict[str, Any]:
        """创建带变化的最小配置"""
        random.seed(seed)
        
        conv_type = random.choice(["DpConv", "DWSepConv"])
        channels = random.choice([8, 16, 24, 32])
        kernel_size = random.choice([3, 5, 7])
        activation = random.choice(["ReLU6", "LeakyReLU", "Swish"])
        
        return {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": conv_type,
                            "kernel_size": kernel_size,
                            "stride": 1,
                            "expansion": 1,
                            "has_se": False,
                            "se_ratio": 0,
                            "activation": activation,
                            "skip_connection": False
                        }
                    ],
                    "channels": channels
                }
            ]
        }
    
    def _create_random_minimal_config(self, direction: str) -> Dict[str, Any]:
        """创建完全随机的最小配置"""
        num_stages = random.choice([1, 2])
        
        stages = []
        for i in range(num_stages):
            conv_type = random.choice(["DpConv", "DWSepConv", "MBConv"])
            channels = random.choice([8, 12, 16, 20])
            kernel_size = random.choice([3, 5])
            stride = 2 if i > 0 else 1
            
            stages.append({
                "blocks": [
                    {
                        "type": conv_type,
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "expansion": random.choice([2, 3]) if conv_type == "MBConv" else 1,
                        "has_se": False,
                        "se_ratio": 0,
                        "activation": random.choice(["ReLU6", "Swish"]),
                        "skip_connection": False
                    }
                ],
                "channels": channels
            })
        
        return {
            "input_channels": self.dataset_info["channels"],
            "num_classes": self.dataset_info["num_classes"],
            "quant_mode": direction,
            "stages": stages
        }
    
    # ========== 降级策略 ==========
    
    def apply_stage_reduction(self, config: Dict[str, Any], 
                              current_memory: float,
                              dup_modification: bool = False) -> Optional[Dict[str, Any]]:
        """策略1: 减少stage数量"""
        print(f"Strategy 1: apply_stage_reduction (memory: {current_memory:.2f}MB)")
        sys.stdout.flush()
        
        if current_memory <= self.max_memory and not dup_modification:
            return None
        
        original_stage_count = len(config.get("stages", []))
        if original_stage_count <= 1:
            return None
        
        print("🔧 策略1: 减少stage数量")
        temp_config = config.copy()
        
        for stage_count in range(original_stage_count - 1, 0, -1):
            temp_config["stages"] = temp_config["stages"][:stage_count]
            memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
            
            if memory_ok:
                print(f"✅ Stage: {original_stage_count} → {stage_count}, "
                      f"内存: {new_memory:.2f}MB")
                return temp_config
        
        return None
    
    def apply_conv_type_optimization(self, config: Dict[str, Any], 
                                     current_memory: float,
                                     dup_modification: bool = False) -> Optional[Dict[str, Any]]:
        """策略2: 优化卷积类型"""
        print(f"Strategy 2: apply_conv_type_optimization (memory: {current_memory:.2f}MB)")
        sys.stdout.flush()
        
        if current_memory <= self.max_memory and not dup_modification:
            return None
        
        print("🔧 策略2: 优化卷积类型")
        # 内存排序: SeDpConv < DpConv < DWSepConv < SeSepConv < MBConv
        conv_type_priority = ["SeDpConv", "DpConv", "DWSepConv", "SeSepConv", "MBConv"]
        
        temp_config = json.loads(json.dumps(config))  # 深拷贝
        
        # 从后往前处理stage
        for stage_idx in range(len(temp_config.get("stages", [])) - 1, -1, -1):
            stage = temp_config["stages"][stage_idx]
            
            for block_idx in range(len(stage.get("blocks", [])) - 1, -1, -1):
                block = stage["blocks"][block_idx]
                current_type = block.get("type", "MBConv")
                
                if current_type == "SeDpConv":
                    continue
                
                try:
                    current_priority = conv_type_priority.index(current_type)
                except ValueError:
                    current_priority = len(conv_type_priority) - 1
                
                # 尝试更小内存的类型
                for smaller_type in conv_type_priority[:current_priority]:
                    original_block = block.copy()
                    original_channels = stage["channels"]

                    block["type"] = smaller_type
                    
                    # 调整属性
                    if smaller_type == "SeDpConv":
                        block["has_se"] = True
                        block["se_ratio"] = 0.25
                        block["skip_connection"] = False
                        block["expansion"] = 1

                        # 修复通道约束
                        if stage_idx == 0 and block_idx == 0:
                            temp_config["stages"][stage_idx]["channels"] = \
                                self.dataset_info["channels"]
                        elif stage_idx > 0:
                            prev_channels = temp_config["stages"][stage_idx - 1]["channels"]
                            temp_config["stages"][stage_idx]["channels"] = prev_channels
                            
                    elif smaller_type in ["SeSepConv"]:
                        block["has_se"] = True
                        block["se_ratio"] = 0.25
                        block["skip_connection"] = False
                        
                    elif smaller_type == "DpConv":
                        block["has_se"] = False
                        block["se_ratio"] = 0
                        block["skip_connection"] = False
                        block["expansion"] = 1
                    
                    elif smaller_type == "DWSepConv":
                        block["skip_connection"] = True if block_idx == 0 else False
                        block["expansion"] = 1
                        if block.get("has_se"):
                            block["has_se"] = False
                            block["se_ratio"] = 0
                    
                    # 检查是否满足内存要求
                    try:
                        memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
                        if memory_ok:
                            print(f"✅ Stage{stage_idx}-Block{block_idx}: "
                                  f"{current_type} → {smaller_type}")
                            return temp_config
                        
                    except Exception as e:
                        print(f"⚠️  尝试 {smaller_type} 失败: {e}")
                        block.update(original_block)
                        temp_config["stages"][stage_idx]["channels"] = original_channels
                        continue
        
        return None
    
    def apply_channel_reduction(self, config: Dict[str, Any], 
                                current_memory: float,
                                dup_modification: bool = False) -> Optional[Dict[str, Any]]:
        """策略3: 减少通道数"""
        print(f"Strategy 3: apply_channel_reduction (memory: {current_memory:.2f}MB)")
        sys.stdout.flush()
        
        if current_memory <= self.max_memory and not dup_modification:
            return None
        
        print("🔧 策略3: 减少通道数")
        
        for stage_idx in range(len(config.get("stages", [])) - 1, -1, -1):
            stage = config["stages"][stage_idx]
            original_channels = stage["channels"]
            
            # 检查是否包含SeDpConv
            has_sedpconv = any(b.get("type") == "SeDpConv" 
                              for b in stage.get("blocks", []))

            if has_sedpconv:
                # SeDpConv stage的channels必须保持特定值
                if stage_idx == 0:
                    required_channels = self.dataset_info["channels"]
                else:
                    required_channels = config["stages"][stage_idx - 1]["channels"]
                
                if stage["channels"] != required_channels:
                    print(f"🔧 Stage{stage_idx} (SeDpConv): "
                          f"强制 channels = {required_channels}")
                    stage["channels"] = required_channels
                
                print(f"⏭️  Stage{stage_idx} 包含SeDpConv，跳过通道缩减")
                continue
            
            for reduction_factor in [0.75, 0.5, 0.25]:
                new_channels = max(8, int(original_channels * reduction_factor))
                if new_channels == stage["channels"]:
                    continue
                
                stage["channels"] = new_channels

                try:
                    memory_ok, new_memory, _ = self.check_memory_fn(config)
                    
                    if memory_ok:
                        print(f"✅ Stage{stage_idx}通道: {original_channels} → {new_channels}")
                        return config
                except Exception as e:
                    print(f"⚠️  缩减失败: {e}")
                    stage["channels"] = original_channels
                    continue
            
            stage["channels"] = original_channels
        
        return None
    
    def apply_block_reduction(self, config: Dict[str, Any], 
                              current_memory: float,
                              dup_modification: bool = False) -> Optional[Dict[str, Any]]:
        """策略4: 减少block数量"""
        print(f"Strategy 4: apply_block_reduction (memory: {current_memory:.2f}MB)")
        sys.stdout.flush()
        
        if current_memory <= self.max_memory and not dup_modification:
            return None
        
        print("🔧 策略4: 减少block数量")
        temp_config = json.loads(json.dumps(config))
        
        for stage_idx in range(len(temp_config["stages"]) - 1, -1, -1):
            stage = temp_config["stages"][stage_idx]
            original_block_count = len(stage["blocks"])
            
            if original_block_count <= 1:
                # 如果只有1个block，尝试删除整个stage（保留第一个stage）
                if stage_idx > 0:
                    print(f"🔄 Stage{stage_idx} 只有1个block，删除整个stage")
                    temp_config["stages"].pop(stage_idx)
                    try:
                        memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
                        if memory_ok:
                            print(f"✅ 删除Stage{stage_idx}后内存达标: "
                                  f"{new_memory:.2f}MB")
                            return temp_config
                    except Exception as e:
                        print(f"⚠️  删除Stage{stage_idx}失败: {e}")
                        temp_config["stages"].insert(stage_idx, stage)
                continue
            
            # 尝试减少block数量
            for new_block_count in range(original_block_count - 1, 0, -1):
                stage["blocks"] = stage["blocks"][:new_block_count]
                try:
                    memory_ok, new_memory, _ = self.check_memory_fn(temp_config)
                    if memory_ok:
                        print(f"✅ Stage{stage_idx} block: "
                              f"{original_block_count} → {new_block_count}")
                        return temp_config
                
                except Exception as e:
                    print(f"⚠️  减少block失败: {e}")
                    continue
        
        return None