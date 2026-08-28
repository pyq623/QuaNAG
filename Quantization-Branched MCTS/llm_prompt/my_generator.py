# /root/unified_nas/llm_prompt/my_generator.py
"""
统一的LLM配置生成器
支持HAR和Image任务的模型配置生成

核心特性:
- 完整的对话记录系统
- 自动适配task_type
- 智能配置验证和修复
- 内存约束自动缩减
"""

import json
import re
import uuid
from typing import Dict, Any, List, Optional
import random


class LLMConfigGenerator:
    """
    LLM配置生成器 - 统一版
    
    负责与LLM交互生成模型配置，支持HAR和Image任务
    """
    
    def __init__(self, search_space: Dict[str, Any], constraint: Dict[str, Any], 
                 dataset_name: str, task_type: str = None):
        """
        初始化LLM配置生成器
        
        Args:
            search_space: 搜索空间定义
            constraint: 约束条件（包含max_peak_memory等）
            dataset_name: 数据集名称
            task_type: 任务类型 ('har' 或 'image'，可选，会从dataset_info推断）
        """
        self.search_space = search_space
        self.constraint = constraint
        self.dataset_name = dataset_name
        self.global_insights = []  # 存储跨分支的成功经验
        
        # ✅ 获取数据集信息
        from data import get_data_loader
        
        # 推断task_type
        if task_type is None:
            # 根据数据集名称推断（HAR数据集通常有特定名称）
            har_datasets = ['MMAct', 'Wharf', 'UTD-MHAD', 'USCHAD', 'Mhealth']
            task_type = 'har' if dataset_name in har_datasets else 'image'
        
        self.task_type = task_type
        
        # 获取数据加载器和数据集信息
        loader = get_data_loader(self.task_type)
        self.dataset_info = loader.get_dataset_info(dataset_name)
        
        # ✅ 确保dataset_info包含task_type
        self.dataset_info['task_type'] = self.task_type
        
        print(f"🔧 LLM配置生成器初始化:")
        print(f"   - 数据集: {dataset_name}")
        print(f"   - 任务类型: {self.task_type}")
        print(f"   - 输入通道: {self.dataset_info['channels']}")
        if self.task_type == 'har':
            print(f"   - 时间步长: {self.dataset_info['time_steps']}")
        else:
            print(f"   - 图像尺寸: {self.dataset_info['height']}×{self.dataset_info['width']}")
        print(f"   - 类别数: {self.dataset_info['num_classes']}")
        
    def init_dialogue_record(self, parent_node: Any, direction: str) -> Dict[str, Any]:
        """
        初始化对话记录结构
        
        Args:
            parent_node: 父节点
            direction: 扩展方向（量化模式）
        
        Returns:
            对话记录字典
        """
        record = {
            "dialogue_id": f"gen_{uuid.uuid4().hex[:8]}",
            "mcts_trajectory_ids": [],
            "mcts_context": {
                "parent_id": parent_node.node_id,
                "expansion_direction": direction,
                "parent_reward": parent_node.average_reward,
                "task_type": self.task_type  # ✅ 记录任务类型
            },
            "messages": []
        }
        
        # 回溯轨迹ID
        temp_node = parent_node
        path_ids = []
        while temp_node:
            path_ids.append(temp_node.node_id)
            if temp_node.parent:
                temp_node = temp_node.parent
            else:
                break
        record["mcts_trajectory_ids"] = list(reversed(path_ids))
        
        # 预加载System Prompt
        system_content = self._build_system_prompt(direction, self.global_insights)
        record["messages"].append({
            "role": "system",
            "content": system_content
        })
        
        return record
    
    def generate_config_with_context(self, 
                                    dialogue_record: Dict[str, Any],
                                    parent_config: Dict[str, Any], 
                                    direction: str,
                                    parent_performance: Dict[str, Any],
                                    global_insights: Dict = None,
                                    memory_feedback: str = None) -> Dict[str, Any]:
        """
        基于上下文生成新配置
        
        Args:
            dialogue_record: 引用传递，直接修改
            parent_config: 父配置
            direction: 量化方向
            parent_performance: 父节点性能
            global_insights: 全局经验
            memory_feedback: 内存反馈
        
        Returns:
            生成的配置
        """
        
        # 构建提示词
        system_prompt = self._build_system_prompt(direction, global_insights)
        
        # ✅ 只在第一次添加system prompt
        if len(dialogue_record["messages"]) == 0 or \
           dialogue_record["messages"][0]["role"] != "system":
            dialogue_record["messages"].append({
                "role": "system", 
                "content": system_prompt
            })
        
        # 使用当前的内存反馈构建提示词
        human_prompt = self._build_human_prompt(
            parent_config, direction, parent_performance, memory_feedback
        )
        dialogue_record["messages"].append({
            "role": "user",
            "content": human_prompt
        })
        
        try:
            print(f"🔍 系统提示词长度: {len(system_prompt)}")
            print(f"🔍 用户提示词长度: {len(human_prompt)}")
            
            # ✅ 使用便捷函数调用LLM
            from utils.llm_utils import call_llm_with_messages
            response = call_llm_with_messages(system_prompt, human_prompt)
            
            print(f"🔍 LLM响应长度: {len(response)}")
            
            # 记录Assistant消息
            dialogue_record["messages"].append({
                "role": "assistant",
                "content": response
            })
            
            # 解析响应并提取JSON配置
            new_config = self._parse_llm_response(response)
            
            # ✅ 验证配置的有效性并自动修复（记录修复过程）
            validated_config = self._validate_and_fix_config(
                new_config, direction, dialogue_record
            )
            
            if validated_config:
                return validated_config
            else:
                # ✅ 记录配置验证失败
                dialogue_record["messages"].append({
                    "role": "user",
                    "content": "--- VALIDATION FAILED ---\n**ERROR:** Configuration validation failed.\n**Fallback:** Using base config generator."
                })
                print("LLM生成的配置无效，使用智能变异")
                base_config = self._generate_base_config(direction, dialogue_record)
                return base_config
                
        except Exception as e:
            print(f"LLM配置生成失败: {e}，使用智能变异")
            # ✅ 记录LLM异常
            dialogue_record["messages"].append({
                "role": "user",
                "content": f"--- LLM EXCEPTION ---\n**ERROR:** {str(e)}\n**Fallback:** Using base config generator."
            })
            base_config = self._generate_base_config(direction, dialogue_record)
            return base_config
    
    def _build_system_prompt(self, direction: str, global_insights: Dict = None) -> str:
        """
        构建系统提示词
        
        Args:
            direction: 量化方向
            global_insights: 全局经验
        
        Returns:
            系统提示词字符串
        """
        # 构建经验部分
        insights_text = ""

        if global_insights:
            useful_insights = []
            
            # 提取与当前方向相关的经验
            for insight_key, insight_data in global_insights.items():
                if insight_key.startswith("direction_"):
                    dir_name = insight_key.replace("direction_", "")
                    success_rate = insight_data.get('success_rate', 0)
                    avg_reward = insight_data.get('average_reward', 0)
                    
                    # 只显示成功率较高的经验
                    if success_rate > 0.6:
                        useful_insights.append(
                            f"方向 '{dir_name}': 成功率 {success_rate:.1%}, "
                            f"平均奖励 {avg_reward:.3f}"
                        )
            
            if useful_insights:
                insights_text = f"""
                GLOBAL EXPERIENCE INSIGHTS (from successful search branches):
                {chr(10).join(f"• {insight}" for insight in useful_insights)}
                """

        system_prompt = f"""
        You are a neural network architecture design expert. Your task is to generate improved network configurations based on given constraints and insights.
        {insights_text}

        **Conv Type in the Search Space:**
            1. DWSepConvBlock: Depthwise separable convolution (Depthwise + Pointwise) structure with skip connection support.
            2. MBConvBlock: Inverted residual structure (expansion convolution + Depthwise + SE module + Pointwise) with skip connection support.
            3. DpConvBlock: Pure depthwise convolution (Depthwise + Pointwise) structure without SE module or skip connections.
            4. SeSepConvBlock: Depthwise separable convolution with SE module (Depthwise + SE + Pointwise) structure.
            5. SeDpConvBlock: Depthwise convolution with SE module (Depthwise + SE) structure without Pointwise convolution.

        **CRITICAL: Memory Consumption Ranking (Single Stage, Single Block Comparison):**
        """
        
        # ✅ 根据task_type和数据集选择示例
        if self.task_type == 'har':
            # HAR任务：根据数据集选择
            if self.dataset_name == "MMAct":
                conv_examples = """
                LIGHTEST → HEAVIEST (MMAct):
                1. SeDpConv:   ~1.6 MB  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
                2. DpConv:     ~2.7 MB  (Depthwise + Pointwise, no SE)
                3. DWSepConv:  ~3.3 MB  (Depthwise + Pointwise + optional skip)
                4. SeSepConv:  ~3.3 MB  (Depthwise + SE + Pointwise)
                5. MBConv:     ~7.7 MB  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
                """
            elif self.dataset_name == "Wharf":
                conv_examples = """
                LIGHTEST → HEAVIEST (Wharf):
                1. SeDpConv:   ~0.7 MB  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
                2. DpConv:     ~1.2 MB  (Depthwise + Pointwise, no SE)
                3. DWSepConv:  ~1.4 MB  (Depthwise + Pointwise + optional skip)
                4. SeSepConv:  ~1.4 MB  (Depthwise + SE + Pointwise)
                5. MBConv:     ~3.3 MB  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
                """
            elif self.dataset_name == "UTD-MHAD":
                conv_examples = """
                LIGHTEST → HEAVIEST (UTD-MHAD):
                1. SeDpConv:   ~1.3 MB  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
                2. DpConv:     ~2.2 MB  (Depthwise + Pointwise, no SE)
                3. DWSepConv:  ~2.7 MB  (Depthwise + Pointwise + optional skip)
                4. SeSepConv:  ~2.7 MB  (Depthwise + SE + Pointwise)
                5. MBConv:     ~6.2 MB  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
                """
            elif self.dataset_name == "USCHAD":
                conv_examples = """
                LIGHTEST → HEAVIEST (USCHAD):
                1. SeDpConv:   ~4.4 MB  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
                2. DpConv:     ~7.3 MB  (Depthwise + Pointwise, no SE)
                3. DWSepConv:  ~8.8 MB  (Depthwise + Pointwise + optional skip)
                4. SeSepConv:  ~8.8 MB  (Depthwise + SE + Pointwise)
                5. MBConv:     ~20.5 MB  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
                """
            elif self.dataset_name == "Mhealth":
                conv_examples = """
                LIGHTEST → HEAVIEST (Mhealth):
                1. SeDpConv:   ~8.4 MB  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
                2. DpConv:     ~14.1 MB  (Depthwise + Pointwise, no SE)
                3. DWSepConv:  ~16.9 MB  (Depthwise + Pointwise + optional skip)
                4. SeSepConv:  ~16.9 MB  (Depthwise + SE + Pointwise)
                5. MBConv:     ~39.3 MB  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
                """
            else:
                # 默认HAR示例
                conv_examples = """
                LIGHTEST → HEAVIEST:
                1. SeDpConv  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
                2. DpConv  (Depthwise + Pointwise, no SE)
                3. DWSepConv  (Depthwise + Pointwise + optional skip)
                4. SeSepConv  (Depthwise + SE + Pointwise)
                5. MBConv  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
                """
        else:
            # Image任务：通用示例
            conv_examples = """
            LIGHTEST → HEAVIEST:
            1. SeDpConv  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
            2. DpConv  (Depthwise + Pointwise, no SE)
            3. DWSepConv  (Depthwise + Pointwise + optional skip)
            4. SeSepConv  (Depthwise + SE + Pointwise)
            5. MBConv  (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
            """
        
        system_prompt += conv_examples
        system_prompt += f"""

        **Quantization Modes (IMPORTANT):**
            - none: No quantization - standard FP32 model (baseline)
            - static: Post-training static quantization - applies INT8 quantization after training (fast but may lose accuracy)
            - qat: Quantization-Aware Training - simulates quantization effects during training to improve accuracy after quantization
            - qaft: Quantization-Aware Fine-Tuning - fine-tunes a pre-trained model with quantization awareness to recover accuracy lost during quantization
            * RECOMMENDED for best accuracy-efficiency trade-off
            * Quantization may yield dramatic degration or slight degration in accuracy depending on model architecture.
            • Quantization modes (static/qat/qaft) ALWAYS reduce model memory to EXACTLY 1/4 of FP32 size (INT8 = 8 bits vs FP32 = 32 bits)
            • Example: If FP32 model uses 20MB, quantized version uses 5MB
            • When designing for quantized modes, you can use 4x larger architectures!
            • When designing for 'none' mode, be extremely conservative with memory, must fit within the given memory constraint.

        **Important Notes:**
            - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
            - "MBConv" is only different from "DWSeqConv" when expansion > 1, otherwise they are the same block.
            - If the type of a convolution block is "SeDpConv", then the `in_channels` and `out_channels` of this convolution block must be equal. This means that: - The `out_channels` of the previous convolution block must be equal to both the `in_channels` and `out_channels` of "SeDpConv".
            - If "SeDpConv" is a block in the first stage, its `channels` should be equal to `input_channels`, otherwise an error will be reported.
            
        Please ensure the configurations adhere to the defined search space constraints.
        """
        return system_prompt
    
    def _build_human_prompt(self, parent_config: Dict[str, Any], 
                          direction: str,
                          parent_performance: Dict[str, Any],
                          memory_feedback: str = None) -> str:
        """
        构建用户提示词
        
        Args:
            parent_config: 父配置
            direction: 量化方向
            parent_performance: 父节点性能
            memory_feedback: 内存反馈
        
        Returns:
            用户提示词字符串
        """
        channels = self.dataset_info['channels']
        num_classes = self.dataset_info['num_classes']
        parent_config_str = json.dumps(parent_config, indent=2) if parent_config else \
                           "No parent configuration available"
        
        max_peak_memory = self.constraint["max_peak_memory"]
        avg_reward = parent_performance.get('average_reward', 0)
        avg_reward = f"{avg_reward:.4f}"
        visit_count = parent_performance.get('visit_count', 0)
        directions_explored = parent_performance.get('directions_explored', [])

        # 添加内存反馈到提示词
        memory_feedback_section = ""
        if memory_feedback:
            memory_feedback_section = f"""
            MEMORY FEEDBACK (Important!):
            {memory_feedback}

            Please generate a configuration that strictly adheres to the memory constraint.
            """

        # ✅ 根据task_type构建输入信息
        if self.task_type == 'har':
            time_steps = self.dataset_info['time_steps']
            input_info = f"Input: {channels} channels, {time_steps} time steps"
        else:  # image
            height = self.dataset_info['height']
            width = self.dataset_info['width']
            input_info = f"Input: {channels} channels, {height}×{width} pixels"

        human_prompt = f"""
        Parent Network Configuration:
        {parent_config_str}

        Parent Network Performance:
        - Average Reward: {avg_reward}
        - Visit Count: {visit_count}
        - Directions Explored: {directions_explored}

        Optimization Direction: {direction}
        (e.g., 'none' for origianl model without quantization, 'static' for static quantization, 'qat' for quantization-aware training, 'qaft' for Quantization-Aware Fine-Tuning.)

        {memory_feedback_section}

        TASK: Generate a new improved network configuration that:
        1. Uses the specified direction: {direction}
        2. Learns from the global experience insights above  
        3. Explores novel but promising architectural changes
        4. Maintains compatibility with the search space constraints, which is a must!
        5. Optimization direction is the specified quantization mode for this generation. The generated configured quantization mode must be the optimization direction.

        CONSTRAINTS (Search Space):
        Max memory: {max_peak_memory} MB
        {input_info}
        Output: {num_classes} classes

        SEARCH SPACE:
        {json.dumps(self.search_space, indent=2)}

        Please generate a new improved network configuration in valid JSON format. 
        The config you generated should have the same input channels and num classes in the example.
        For example:
        ```json
        {{
                "input_channels": {channels},  
                "num_classes": {num_classes},
                "quant_mode": "none",
                "stages": [
                    {{
                        "blocks": [
                            {{
                                "type": "DWSepConv",
                                "kernel_size": 3,
                                "expansion": 3,
                                "has_se": false,
                                "se_ratios": 0,
                                "skip_connection": false,
                                "stride": 1,
                                "activation": "ReLU6"
                            }}
                        ],
                        "channels": 8
                    }},
                    {{
                        "blocks": [
                            {{
                                "type": "MBConv",
                                "kernel_size": 3,
                                "expansion": 4,
                                "has_se": true,
                                "se_ratios": 0.25,
                                "skip_connection": true,
                                "stride": 2,
                                "activation": "Swish"
                            }}
                        ],
                        "channels": 16
                    }}
                ]
            }}
        ```

        Return ONLY the JSON configuration."""
        
        return human_prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析LLM响应，提取JSON配置
        
        Args:
            response_text: LLM响应文本
        
        Returns:
            解析后的配置字典
        """
        try:
            # 尝试直接解析JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，尝试解析为代码块
                code_block_match = re.search(
                    r'```(?:json)?\s*(.*?)\s*```', 
                    response_text, 
                    re.DOTALL
                )
                if code_block_match:
                    json_str = code_block_match.group(1)
                    return json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in response")
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON解析失败: {e}")
            raise
    
    def _validate_and_fix_config(self, config: Dict[str, Any], target_direction: str,
                                 dialogue_record: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        验证配置的有效性并自动修复
        
        Args:
            config: 待验证的配置
            target_direction: 目标量化方向
            dialogue_record: 对话记录（可选）
        
        Returns:
            验证并修复后的配置，失败则返回None
        """
        # ✅ 收集所有修复操作
        fixes_applied = []

        required_fields = ['input_channels', 'num_classes', 'quant_mode', 'stages']
        
        # 检查必需字段
        for field in required_fields:
            if field not in config:
                print(f"Missing required field: {field}")
                return None
        
        # 检查阶段结构
        if not isinstance(config['stages'], list):
            print("Stages must be a list")
            return None
        
        # 强制修复量化模式
        if config.get('quant_mode') != target_direction:
            print(f"Auto-fixing: quant_mode should be '{target_direction}', "
                  f"but got '{config.get('quant_mode')}'. "
                  f"Setting to '{target_direction}'")
            old_mode = config.get('quant_mode')
            config['quant_mode'] = target_direction
            fix_msg = f"Auto-fix: quant_mode '{old_mode}' → '{target_direction}'"
            fixes_applied.append(fix_msg)
        
        # 修复和验证每个阶段
        input_channels = config['input_channels']
        current_channels = input_channels
        
        for stage_idx, stage in enumerate(config['stages']):
            if 'channels' not in stage or 'blocks' not in stage:
                print("Invalid stage structure")
                return None
            
            if not isinstance(stage['blocks'], list) or len(stage['blocks']) == 0:
                print("Blocks must be a non-empty list")
                return None
            
            stage_channels = stage['channels']
            
            # 验证和修复每个block
            for block_idx, block in enumerate(stage['blocks']):
                block_fixes = self._validate_and_fix_block(
                    block, stage_idx, block_idx, 
                    current_channels, stage_channels, input_channels
                )
                
                if block_fixes is False:
                    return None
                
                if block_fixes:
                    fixes_applied.extend(block_fixes)

                # 处理SeDpConv的通道修复
                if block['type'] == 'SeDpConv' and '_stage_channels_fixed' in block:
                    fixed_channels = block['_stage_channels_fixed']
                    old_channels = stage['channels']
                    stage['channels'] = fixed_channels
                    stage_channels = fixed_channels
                    fix_msg = (f"SeDpConv channel fix: stage {stage_idx} "
                             f"channels {old_channels} → {fixed_channels}")
                    print(fix_msg)
                    fixes_applied.append(fix_msg)
                    del block['_stage_channels_fixed']
                    current_channels = fixed_channels
                else:
                    if block['type'] == 'SeDpConv':
                        current_channels = current_channels
                    else:
                        current_channels = stage['channels']
        
        # ✅ 内存检查和自动缩减
        max_memory = self.constraint.get("max_peak_memory", None)
        if max_memory is not None:
            max_memory = float(max_memory)  # 转换为MB
            print(f"max_memory: {max_memory} MB")
            
            from models import CandidateModel
            from utils import calculate_memory_usage
            
            candidate = CandidateModel(config=config, task_type=self.task_type)
            model = candidate.build_model()
            
            # ✅ 根据task_type构建输入尺寸
            if self.task_type == 'har':
                input_size = (64, self.dataset_info['channels'], 
                            self.dataset_info['time_steps'])
            else:  # image
                input_size = (1, self.dataset_info['channels'], 
                            self.dataset_info['height'], 
                            self.dataset_info['width'])
            
            memory_info = calculate_memory_usage(model, input_size=input_size, device='cpu')
            memory_usage = memory_info["total_memory_MB"]
            
            # 考虑量化压缩
            if config.get('quant_mode') in ["static", "qat", "qaft"]:
                memory_usage /= 4.0
            
            # 如果超过限制，自动缩减
            if memory_usage > max_memory:
                print(f"⚠️ 配置内存 {memory_usage:.2f}MB 超过 {max_memory}MB，自动缩减...")
                fixes_applied.append(
                    f"Memory reduction: {memory_usage:.2f}MB → target {max_memory}MB"
                )
                config = self._auto_reduce_memory(
                    config, memory_usage, max_memory, fixes_applied
                )
        
        # ✅ 记录所有修复操作
        if fixes_applied and dialogue_record:
            fixes_summary = "\n".join(f"  - {fix}" for fix in fixes_applied)
            feedback_msg = f"""
            --- AUTO-CORRECTION APPLIED ---
            **Total fixes:** {len(fixes_applied)}

            **Corrections made:**
            {fixes_summary}

            **Result:** Configuration has been automatically corrected and is now valid.
            """
            dialogue_record["messages"].append({
                "role": "user",
                "content": feedback_msg
            })
        
        return config
    
    def _auto_reduce_memory(self, config: Dict[str, Any], current_memory: float, 
                       target_memory: float, fixes_applied: list = None) -> Dict[str, Any]:
        """智能缩减配置以满足内存约束 - 遵循 search_space 约束
        Args:
            config: 配置
            current_memory: 当前内存
            target_memory: 目标内存
            fixes_applied: 修复记录列表 （可选）
        """
        
        # 计算需要缩减的比例
        reduction_ratio = target_memory / current_memory * 0.9  # 留10%余量
        
        print(f"🔧 目标缩减比例: {reduction_ratio:.2f}")
        
        # ✅ 修复：从 search_space 获取合法的通道数选项，并确保转为整数
        valid_channels = self.search_space.get('channels', [8, 16, 24, 32])
        
        # ✅ 关键修复： 确保所有通道数都是整数类型
        valid_channels = sorted([int(c) for c in valid_channels])  # 转为 int 并排序
        print(f"📋 合法通道数选项: {valid_channels}")
        
        # 获取 input_channels（用于 SeDpConv 特殊处理）
        input_channels = config.get('input_channels', 9)
        
        # 策略1: 智能缩减所有 stage 的通道数（选择 search_space 中最接近的值）
        for stage_idx, stage in enumerate(config['stages']):
            
            old_channels = int(stage['channels'])  # ✅ 确保是整数
            
            # 计算目标通道数
            target_channels = int(old_channels * reduction_ratio)
            
            # ✅ 特殊处理：检查这个stage是否包含SeDpConv
            has_sedpconv = any(block.get('type') == 'SeDpConv' for block in stage.get('blocks', []))
            
            if has_sedpconv:
                # SeDpConv 要求 in_channels == out_channels
                if stage_idx == 0:
                    # 第一个 stage 必须等于 input_channels
                    new_channels = input_channels
                    print(f"  - Stage {stage_idx} (SeDpConv): 保持 {new_channels} (=input_channels)")
                else:
                    # 其他stage保持与前一个stage相同
                    prev_stage_channels = config['stages'][stage_idx - 1]['channels']
                    new_channels = int(prev_stage_channels)
                    print(f"  - Stage {stage_idx} (SeDpConv): 保持 {new_channels} (=前stage输出)")

            else:
                # 普通卷积：从合法选项中找到最接近目标的值 （但不超过原值）
                new_channels = self._find_closest_valid_channel(
                    target_channels, valid_channels, max_value=old_channels
                )
                print(f"  - Stage {stage_idx} 通道: {old_channels} → {new_channels}")
            
            if new_channels != old_channels and fixes_applied is not None:
                fixes_applied.append(f"Stage{stage_idx} channels: {old_channels} → {new_channels}")
            
            stage['channels'] = new_channels
        
        # 策略2: 如果还不够，减少 blocks 数量（优先移除非SeDpConv的blocks）
        if reduction_ratio < 0.7:
            print(f"⚠️ 缩减比例 < 0.7，开始移除blocks...")
            for stage in reversed(config['stages']):  # 从后往前
                if len(stage['blocks']) > 1:
                    # 优先移除非 SeDpConv 的 block
                    for i in range(len(stage['blocks']) - 1, -1, -1):
                        if stage['blocks'][i].get('type') != 'SeDpConv':
                            removed = stage['blocks'].pop(i)
                            fix_msg = f"Removed block: {removed['type']}"
                            print(f"  - 移除 block : {removed['type']}")
                            if fixes_applied is not None:
                                fixes_applied.append(fix_msg)
                            break
                    else:
                        # 如果都是SeDpConv，移除最后一个
                        removed = stage['blocks'].pop()
                        fix_msg = f"Removed block: {removed['type']}"
                        print(f"  - 移除 block : {removed['type']}")
                        if fixes_applied is not None:
                            fixes_applied.append(fix_msg)
                    break  # 每次只移除一个
        
        # 策略3: 如果还不够，减少stages （从最后开始移除，避免移除包含SeDpConv的第一个stage）
        if reduction_ratio < 0.5 and len(config['stages']) > 2:
            print(f"⚠️ 缩减比例 < 0.5，开始移除stage...")
            # 从最后一个stage开始移除（保护第一个stage）
            removed_stage = config['stages'].pop()
            fix_msg = f"Removed stage (channels: {removed_stage['channels']})"
            print(f"  - 移除整个stage (原通道数: {removed_stage['channels']})")
            if fixes_applied is not None:
                fixes_applied.append(fix_msg)
        
        # 策略4: 如果还是不够，进一步缩减expansion
        if reduction_ratio < 0.4:
            print(f"⚠️ 缩减比例 < 0.4，开始缩减expansion...")
            valid_expansions = self.search_space.get('expansions', [1, 2, 3, 4])
            # ✅ 确保是整数
            valid_expansions = sorted([int(e) for e in valid_expansions])
            min_expansion = min(valid_expansions)
            
            for stage in config['stages']:
                for block in stage['blocks']:
                    current_expansion = int(block.get('expansion', 1))
                    if current_expansion > min_expansion:
                        # 选择比当前小的最大expansion
                        smaller_expansions = [e for e in valid_expansions if e < current_expansion]
                        if smaller_expansions:
                            old_expansion = current_expansion
                            block['expansion'] = max(smaller_expansions)
                            fix_msg = f"Block expansion: {old_expansion} → {block['expansion']}"
                            print(f"  - 缩减block expansion: {old_expansion} → {block['expansion']}")
                            if fixes_applied is not None:
                                fixes_applied.append(fix_msg)
        
        return config

    def _find_closest_valid_channel(self, target: int, valid_options: list, max_value: int = None) -> int:
        """
        从合法选项中找到最接近目标值的通道数
        
        Args:
            target: 目标通道数（整数）
            valid_options: 合法的通道数列表（必须已排序，且都是整数）
            max_value: 最大值限制（不能超过原值）
        
        Returns:
            最接近目标的合法通道数
        """
        # ✅ 确保所有输入都是整数
        target = int(target)
        valid_options = [int(c) for c in valid_options]
        
        if max_value is not None:
            max_value = int(max_value)
            # 过滤掉超过最大值的选项
            valid_options = [c for c in valid_options if c <= max_value]
        
        if not valid_options:
            return 8  # 最小默认值
        
        # 如果目标小于最小合法值，返回最小值
        if target <= valid_options[0]:
            return valid_options[0]
        
        # 如果目标大于最大合法值，返回最大值
        if target >= valid_options[-1]:
            return valid_options[-1]
        
        # 找到最接近的值
        closest = min(valid_options, key=lambda x: abs(x - target))
        return closest
    
    def _validate_and_fix_block(self, block: Dict[str, Any], stage_idx: int, block_idx: int, 
                              current_channels: int, stage_channels: int, input_channels: int) -> bool:
        """验证和修复单个 block 的配置 - 增强版（返回修复列表）
        Returns:
            list: 修复操作列表，如果验证失败返回 False
        """
        fixes = []

        required_block_fields = ['type', 'kernel_size', 'stride', 'expansion', 'has_se', 'se_ratio', 'skip_connection', 'activation']
        
        # 检查必需字段
        for field in required_block_fields:
            if field not in block:
                print(f"Block missing required field: {field}")
                return False
        
        # 验证卷积类型
        valid_conv_types = ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"]
        if block['type'] not in valid_conv_types:
            print(f"Invalid conv type: {block['type']}")
            return False
        
        # 处理 SeDpConv 的特殊约束
        if block['type'] == 'SeDpConv':
            # SeDpConv 要求输入输出通道相等
            if stage_idx == 0 and block_idx == 0:
                # 第一个stage的第一个block，通道数必须等于 input_channels
                if stage_channels != input_channels:
                    fix_msg = f"Auto-fixing: SeDpConv in first block requires channels equal to input_channels ({input_channels}), changing from {stage_channels} to {input_channels}"
                    print(fix_msg)
                    fixes.append(fix_msg)
                    # 直接修改stage的channels
                    block['_stage_channels_fixed'] = input_channels  # 标记需要修复
            else:
                # 其他位置的SeDpConv，通道数必须等于当前输入通道
                if stage_channels != current_channels:
                    fix_msg = f"Auto-fixing: SeDpConv requires channels equal to input channels ({current_channels}), changing from {stage_channels} to {current_channels}"
                    print(fix_msg)
                    fixes.append(fix_msg)
                    # 直接修改stage的channels
                    block['_stage_channels_fixed'] = current_channels  # 标记需要修复
        
        # 验证和修复 SE 模块设置
        if block['type'] in ['SeDpConv', 'SeSepConv']:
            # 这些类型必须启用 SE
            if not block['has_se']:
                fix_msg = f"Auto-fixing: {block['type']} must have has_se = True\n{block['type']}: has_se False → True"
                print(f"Auto-fixing: {block['type']} must have has_se = True")
                fixes.append(fix_msg)
                block['has_se'] = True
            
            if block['se_ratio'] <= 0:
                fix_msg = f"{block['type']}: se_ratio {block['se_ratio']} → 0.25"
                # 设置合理的默认 SE ratio
                fixes.append(fix_msg)
                block['se_ratio'] = 0.25
                print(f"Auto-fixing: {block['type']} must have se_ratio > 0, set to 0.25")
        
        # ✅ DpConv 不支持 SE 模块
        if block['type'] == 'DpConv':
            if block['has_se'] or block['se_ratio'] > 0:
                issue_msg = "DpConv does not support SE module"
                fix_msg = f"{issue_msg}\nDpConv: has_se={block['has_se']}, se_ratio={block['se_ratio']} → False, 0"
                
                print(f"Auto-fixing: {fix_msg}")
                fixes.append(fix_msg)
                
                block['has_se'] = False
                block['se_ratio'] = 0

        # 验证 has_se 和 se_ratio 的一致性
        if block['has_se'] and block['se_ratio'] <= 0:
            fix_msg = f"Auto-fixing: has_se=True but se_ratio={block['se_ratio']}, setting se_ratio=0.25"
            print(fix_msg)
            fixes.append(fix_msg)
            block['se_ratio'] = 0.25
        elif not block['has_se'] and block['se_ratio'] > 0:
            fix_msg = f"Auto-fixing: has_se=False but se_ratio={block['se_ratio']}, setting se_ratio=0"
            print(fix_msg)
            fixes.append(fix_msg)
            block['se_ratio'] = 0
        
        # 验证和修复 skip connection
        if block['type'] in ['DpConv', 'SeDpConv', 'SeSepConv']:
            # 这些类型不支持 skip connection
            if block['skip_connection']:
                fix_msg = f"Auto-fixing: {block['type']} does not support skip_connection\n{block['type']}: skip_connection True → False"
                print(fix_msg)
                fixes.append(fix_msg)
                block['skip_connection'] = False
        
        # 验证 MBConv 和 DWSepConv 的关系
        if block['type'] == 'MBConv' and block['expansion'] == 1:
            fix_msg = f"Auto-fixing: MBConv with expansion=1 is equivalent to DWSepConv\nMBConv expansion=1 → 2 (to differentiate from DWSepConv)"
            # MBConv with expansion=1 实际上就是 DWSepConv
            print(fix_msg)
            fixes.append(fix_msg)
            block['expansion'] = 2
        elif block['type'] == 'DWSepConv' and block['expansion'] > 1:
            fix_msg = f"Auto-fixing: DWSepConv with expansion>1 is equivalent to MBConv\nDWSepConv expansion={block['expansion']} → 1"
            # DWSepConv with expansion>1 实际上就是 MBConv
            print(fix_msg)
            fixes.append(fix_msg)
            block['expansion'] = 1
        
        # 验证 expansion 范围
        if block['expansion'] < 1:
            fix_msg = f"Auto-fixing: expansion cannot be < 1, setting to 1\nexpansion {block['expansion']} → 1"
            print(fix_msg)
            fixes.append(fix_msg)
            block['expansion'] = 1
        
        # 验证 kernel_size
        if block['kernel_size'] not in [1, 3, 5, 7]:
            old_ks = block['kernel_size']
            block['kernel_size'] = 3
            fix_msg = f"kernel_size {old_ks} → 3"
            print(f"Auto-fixing: {fix_msg}")
            block['kernel_size'] = 3
            fixes.append(fix_msg)
        
        # 验证 stride (第一个block可以有stride>1，其他block应该为1)
        if block_idx > 0 and block['stride'] > 1:
            fix_msg = f"Auto-fixing: only first block in stage can have stride>1\nNon-first block stride {block['stride']} → 1"
            print(fix_msg)
            fixes.append(fix_msg)
            block['stride'] = 1
        
        # 验证 activation
        valid_activations = ['ReLU', 'ReLU6', 'Swish', 'HSwish', 'LeakyReLU']
        if block['activation'] not in valid_activations:
            old_act = block['activation']
            print(f"Auto-fixing: invalid activation {block['activation']}, setting to ReLU")
            block['activation'] = 'ReLU'
            fix_msg = f"Auto-fixing: invalid activation {block['activation']}, setting to ReLU\nactivation {old_act} → ReLU"
            fixes.append(fix_msg)
        
        return fixes
    
    def _generate_base_config(self, direction: str, 
                              dialogue_record: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成基础配置（回退方案）
        Args:
            direction: 量化方向
            dialogue_record: 对话记录（可选）
        """
        import random
        print(f"生成基础配置，方向: {direction}")
        
        # 随机化参数
        stage1_channels = random.choice([8, 12, 16])
        stage2_channels = random.choice([16, 24, 32])

        conv_types = ["DWSepConv", "MBConv", "SeSepConv"]
        activations = ["ReLU", "ReLU6", "Swish"]

        # 随机选择第一个stage的卷积类型
        stage1_type = random.choice(conv_types)
        stage1_expansion = 1 if stage1_type == "DWSepConv" else random.choice([2, 3, 4])
        stage1_has_se = random.choice([True, False])
        
        # 随机选择第二个stage的卷积类型
        stage2_type = random.choice(["MBConv", "SeSepConv"])
        stage2_expansion = random.choice([2, 3, 4, 6])

        base_config = {
            "input_channels": self.dataset_info['channels'],
            "num_classes": self.dataset_info['num_classes'],
            "quant_mode": direction,
            "stages": [
                {
                    "blocks": [
                        {
                            "type": stage1_type,
                            "kernel_size": random.choice([3, 5]),
                            "stride": 1,
                            "expansion": stage1_expansion,
                            "has_se": stage1_has_se,
                            "se_ratio": 0.25 if stage1_has_se else 0,
                            "skip_connection": stage1_type in ["DWSepConv", "MBConv"],
                            "activation": random.choice(activations)
                        }
                    ],
                    "channels": stage1_channels
                },
                {
                    "blocks": [
                        {
                            "type": stage2_type,
                            "kernel_size": random.choice([3, 5, 7]),
                            "stride": 2,
                            "expansion": stage2_expansion,
                            "has_se": True,
                            "se_ratio": 0.25,
                            "skip_connection": True,
                            "activation": random.choice(activations)
                        }
                    ],
                    "channels": stage2_channels
                }
            ]
        }

        # ✅ 记录基础配置生成
        if dialogue_record:
            config_summary = f"""
            --- BASE CONFIG GENERATOR ---
            **Fallback Strategy:** Random base configuration

            **Generated:**
            - Stage 1: {stage1_type}, channels={stage1_channels}
            - Stage 2: {stage2_type}, channels={stage2_channels}
            - Quantization: {direction}
            """
            dialogue_record["messages"].append({
                "role": "user",
                "content": config_summary
            })
            
            dialogue_record["messages"].append({
                "role": "assistant",
                "content": f"```json\n{json.dumps(base_config, indent=2)}\n```"
            })
        
        return base_config
    