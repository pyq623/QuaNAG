import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from itertools import combinations
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 添加到父文件夹的路径
from data import get_dataset_info

class MCTSDPOBuilder:
    """从MCTS search_state.json构建DPO数据集"""
    
    def __init__(self, 
                 search_state_paths: List[str],
                 memory_limit: float = 15.0,
                 min_accuracy_threshold: float = 70.0,
                 chosen_accuracy_threshold: float = 90.0):  # 新增！
        """
        Args:
            search_state_paths: MCTS search_state.json 文件路径列表
            memory_limit: 内存限制（MB）
            min_accuracy_threshold: 过滤掉准确率低于此值的 configs（用于rejected）
            chosen_accuracy_threshold: chosen必须达到的准确率阈值
        """
        self.memory_limit = memory_limit
        self.min_accuracy_threshold = min_accuracy_threshold
        self.chosen_accuracy_threshold = chosen_accuracy_threshold  # 新增！
        self.search_state_paths = search_state_paths
        
        # 统计信息
        self.stats = {
            'total_configs': 0,
            'valid_configs': 0,  # memory <= limit
            'invalid_configs': 0,  # memory > limit
            'filtered_low_acc': 0,
            'high_quality_configs': 0  # 新增：满足 chosen 阈值的
        }

        # 加载所有数据
        self.all_configs = []
        self.load_all_data()
        
    def compute_reward(self, config: Dict) -> float:
        """
        计算综合reward
        
        这是核心评估函数：
        - 满足内存：reward = accuracy
        - 超出内存：reward = 0
        """
        accuracy = config['accuracy']
        memory = config['memory_mb']
        
        if memory > self.memory_limit:
            # 非法：惩罚
            reward = 0
        else:
            # 合法：准确率为主
            reward = accuracy
        
        return reward
    
    def load_all_data(self):
        """加载所有MCTS数据"""
        print("\n" + "="*60)
        print("📂 Loading MCTS Data")
        print("="*60)
        
        for path in self.search_state_paths:
            print(f"\nLoading: {path}")
            
            try:
                with open(path) as f:
                    data = json.load(f)
                
                # 提取nodes
                nodes = data.get('nodes', {})
                
                configs_from_file = 0
                for node_id, node_data in nodes.items():
                    # 提取必要信息
                    config = {
                        'node_id': node_id,
                        'config': node_data.get('candidate_config', {}),
                        'accuracy': node_data.get('metrics', {}).get('accuracy', 0),
                        'memory_mb': node_data.get('metrics', {}).get('memory_mb', 0),
                        'quantization_mode': node_data.get('metrics', {}).get('quantization_mode', 'unknown'),
                        'source_file': Path(path).name
                    }
                    
                    # 过滤极低准确率的（可能是训练失败）
                    if config['accuracy'] < self.min_accuracy_threshold:
                        self.stats['filtered_low_acc'] += 1
                        continue
                    
                    self.all_configs.append(config)
                    configs_from_file += 1
                
                print(f"  ✅ Loaded {configs_from_file} configs")
                
            except Exception as e:
                print(f"  ❌ Error loading {path}: {e}")
        
        print(f"\n📊 Total configs loaded: {len(self.all_configs)}")
        print(f"   Filtered (acc < {self.min_accuracy_threshold}%): {self.stats['filtered_low_acc']}")
    
    def analyze_data(self):
        """分析数据分布"""
        print("\n" + "="*60)
        print("📊 Data Distribution Analysis")
        print("="*60)
        
        # 统计内存分布
        valid = [c for c in self.all_configs if c['memory_mb'] <= self.memory_limit]
        invalid = [c for c in self.all_configs if c['memory_mb'] > self.memory_limit]
        
        # 新增：统计高质量configs（满足chosen阈值）
        high_quality = [c for c in valid if c['accuracy'] >= self.chosen_accuracy_threshold]
        
        self.stats['total_configs'] = len(self.all_configs)
        self.stats['valid_configs'] = len(valid)
        self.stats['invalid_configs'] = len(invalid)
        self.stats['high_quality_configs'] = len(high_quality)  # 新增！
        
        print(f"\nMemory constraint: ≤ {self.memory_limit}MB")
        print(f"  Valid (≤ {self.memory_limit}MB): {len(valid)} ({len(valid)/len(self.all_configs)*100:.1f}%)")
        print(f"  Invalid (> {self.memory_limit}MB): {len(invalid)} ({len(invalid)/len(self.all_configs)*100:.1f}%)")
        
        # 新增：显示高质量configs统计
        print(f"\nHigh-quality configs (valid + acc ≥ {self.chosen_accuracy_threshold}%):")
        print(f"  Count: {len(high_quality)} ({len(high_quality)/len(valid)*100:.1f}% of valid)")
        
        if not high_quality:
            print(f"\n⚠️  WARNING: No configs meet chosen threshold ({self.chosen_accuracy_threshold}%)!")
            print(f"   Consider lowering chosen_accuracy_threshold or checking your data.")
        
        # 准确率分布
        all_accs = [c['accuracy'] for c in self.all_configs]
        valid_accs = [c['accuracy'] for c in valid]
        invalid_accs = [c['accuracy'] for c in invalid] if invalid else [0]
        high_quality_accs = [c['accuracy'] for c in high_quality] if high_quality else [0]
        
        print(f"\nAccuracy distribution:")
        print(f"  All configs: {min(all_accs):.2f}% - {max(all_accs):.2f}% (mean: {sum(all_accs)/len(all_accs):.2f}%)")
        if valid_accs:
            print(f"  Valid configs: {min(valid_accs):.2f}% - {max(valid_accs):.2f}% (mean: {sum(valid_accs)/len(valid_accs):.2f}%)")
        if invalid_accs and invalid:
            print(f"  Invalid configs: {min(invalid_accs):.2f}% - {max(invalid_accs):.2f}% (mean: {sum(invalid_accs)/len(invalid_accs):.2f}%)")
        if high_quality_accs and high_quality:
            print(f"  High-quality configs: {min(high_quality_accs):.2f}% - {max(high_quality_accs):.2f}% (mean: {sum(high_quality_accs)/len(high_quality_accs):.2f}%)")
        
        # 内存分布
        all_mems = [c['memory_mb'] for c in self.all_configs]
        print(f"\nMemory distribution:")
        print(f"  Range: {min(all_mems):.2f}MB - {max(all_mems):.2f}MB")
        print(f"  Mean: {sum(all_mems)/len(all_mems):.2f}MB")
        
        # 量化模式分布
        quant_dist = defaultdict(int)
        for c in self.all_configs:
            quant_dist[c['quantization_mode']] += 1
        
        print(f"\nQuantization mode distribution:")
        for mode, count in sorted(quant_dist.items()):
            print(f"  {mode}: {count} ({count/len(self.all_configs)*100:.1f}%)")
        
        return valid, invalid, high_quality  # 新增返回high_quality
    
    def build_dataset(self, 
                     output_path: str = 'dpo_mcts_dataset.jsonl',
                     pairs_per_chosen: int = 2,
                     max_pairs: int = None) -> List[Dict]:
        """
        构建DPO数据集
        
        Args:
            output_path: 输出路径
            pairs_per_chosen: 每个chosen配几个rejected (推荐2)
            max_pairs: 最多生成多少对（None=不限制）
        """
        print("\n" + "="*60)
        print("🔨 Building DPO Dataset")
        print("="*60)
        
        valid, invalid, high_quality = self.analyze_data()  # 修改：接收high_quality
        
        # 修改：检查高质量configs
        if not high_quality:
            print(f"\n❌ No high-quality configs (acc ≥ {self.chosen_accuracy_threshold}%) found!")
            print(f"   Cannot build dataset. Please:")
            print(f"   1. Lower chosen_accuracy_threshold, or")
            print(f"   2. Check if your data has high-accuracy configs")
            return []
        
        pairs = []
        
        # === 策略1: 合法 vs 非法（最重要）===
        print(f"\n📋 Strategy 1: Valid (acc ≥ {self.chosen_accuracy_threshold}%) vs Invalid")
        violation_pairs = self.create_violation_pairs(high_quality, invalid, pairs_per_chosen)  # 修改：用high_quality
        pairs.extend(violation_pairs)
        print(f"  Created {len(violation_pairs)} pairs")
        
        # === 策略2: 高质量 vs 低质量（都合法）===
        print(f"\n📋 Strategy 2: High Quality (acc ≥ {self.chosen_accuracy_threshold}%) vs Lower Quality")
        quality_pairs = self.create_quality_pairs(high_quality, valid, pairs_per_chosen)  # 修改：传入high_quality和valid
        pairs.extend(quality_pairs)
        print(f"  Created {len(quality_pairs)} pairs")
        
        # # === 策略3: 混合策略（1个chosen + 1个超内存 + 1个低质量）===
        # print(f"\n📋 Strategy 3: Mixed (high quality vs invalid + lower quality)")
        # mixed_pairs = self.create_mixed_pairs(high_quality, valid, invalid, pairs_per_chosen)  # 修改
        # pairs.extend(mixed_pairs)
        # print(f"  Created {len(mixed_pairs)} pairs")
        
        # 去重
        pairs = self.deduplicate_pairs(pairs)
        print(f"\n✅ Total unique pairs: {len(pairs)}")
        
        # 限制数量
        if max_pairs and len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)
            print(f"  Sampled {max_pairs} pairs")
        
        # 保存
        self.save_dataset(pairs, output_path)
        
        # 统计
        self.print_statistics(pairs)
        
        return pairs
    
    def create_violation_pairs(self, high_quality: List[Dict], invalid: List[Dict], 
                               pairs_per_chosen: int = 2) -> List[Dict]:
        """策略1: 高质量合法 vs 非法"""
        pairs = []
        
        if not invalid or not high_quality:
            return pairs
        
        # 按准确率排序 high_quality configs
        high_quality_sorted = sorted(high_quality, key=lambda x: x['accuracy'], reverse=True)
        
        # 取所有 high_quality 作为 chosen 候选（因为它们都满足阈值）
        for chosen in high_quality_sorted:
            # 找"诱惑性"的 invalid configs
            # 特点：准确率接近或更高，但内存超限
            tempting = [
                r for r in invalid
                if r['accuracy'] <= chosen['accuracy'] * 1.02  # 准确率至少是 chosen 的80%
                and r['memory_mb'] > self.memory_limit  # 只超15%
            ]
            
            # 如果没有诱惑性的，就随机选一些invalid
            if not tempting:
                tempting = random.sample(invalid, min(pairs_per_chosen, len(invalid)))
            
            # 按准确率排序，选择准确率高的（更诱惑）
            tempting_sorted = sorted(tempting, key=lambda x: x['accuracy'], reverse=True)
            
            # 为这个chosen创建pairs_per_chosen个rejected
            for rejected in tempting_sorted[:pairs_per_chosen]:
                pairs.append(self.create_pair(
                    chosen=chosen,
                    rejected=[rejected],
                    pair_type='violation'
                ))
        
        return pairs
    
    def create_quality_pairs(self, high_quality: List[Dict], all_valid: List[Dict],
                            pairs_per_chosen: int = 2) -> List[Dict]:
        """策略2: 高质量 vs 低质量（都满足内存）"""
        pairs = []
        
        if len(high_quality) < 1:
            return pairs
        
        # 找出低质量但仍满足内存的configs
        # 修改：rejected必须低于chosen阈值
        lower_quality = [c for c in all_valid 
                        if c['accuracy'] <= self.chosen_accuracy_threshold * 1.05]   # 之前选择小于min_accuracy_threshold，改为小于等于100%
        
        if not lower_quality:
            return pairs
        
        # 按准确率排序
        high_quality_sorted = sorted(high_quality, key=lambda x: x['accuracy'], reverse=True)
        lower_quality_sorted = sorted(lower_quality, key=lambda x: x['accuracy'])  # 从低到高
        
        # 为每个high_quality config创建pairs
        for chosen in high_quality_sorted:  # 最多取30个chosen   [:30]
            # 选择准确率明显低于chosen的
            worse_configs = [
                r for r in lower_quality_sorted
                if chosen['accuracy'] - r['accuracy'] > 0  # 至少差5%
            ]
            
            if not worse_configs:
                continue
            
            # 为这个chosen创建pairs_per_chosen个rejected
            for rejected in worse_configs[:pairs_per_chosen]:
                pairs.append(self.create_pair(
                    chosen=chosen,
                    rejected=[rejected],
                    pair_type='quality'
                ))
        
        return pairs
    
    def create_mixed_pairs(self, high_quality: List[Dict], all_valid: List[Dict],
                          invalid: List[Dict], pairs_per_chosen: int = 2) -> List[Dict]:
        """策略3: 高质量 chosen + 2 rejected (1个超内存 + 1个低质量)"""
        pairs = []
        
        if not invalid or len(all_valid) < 2:
            return pairs
        
        # 找出低质量configs
        lower_quality = [c for c in all_valid 
                        if c['accuracy'] < 100]  # chosen_accuracy_threshold 改为100%
        
        if not lower_quality:
            return pairs
        
        # 选择top high_quality configs作为chosen
        high_quality_sorted = sorted(high_quality, key=lambda x: x['accuracy'], reverse=True)
        top_chosen = high_quality_sorted[:min(20, len(high_quality_sorted))]
        
        for chosen in top_chosen:
            rejected_list = []
            
            # 1. 找一个超内存但准确率高的
            tempting_invalid = [
                r for r in invalid
                if r['accuracy'] >= chosen['accuracy'] * 0.92  # 准确率至少是chosen的92%
            ]
            if tempting_invalid:
                rejected_list.append(
                    max(tempting_invalid, key=lambda x: x['accuracy'])
                )
            
            # 2. 找一个满足内存但准确率低的
            worse_valid = [
                r for r in lower_quality
                if r['accuracy'] < chosen['accuracy'] - 8.0  # 低至少8%
            ]
            if worse_valid:
                rejected_list.append(
                    min(worse_valid, key=lambda x: x['accuracy'])
                )
            
            # 确保有2个rejected
            if len(rejected_list) == 2:
                pairs.append(self.create_pair(
                    chosen=chosen,
                    rejected=rejected_list,
                    pair_type='mixed'
                ))
        
        return pairs
    
    def create_pair(self, chosen: Dict, rejected: List[Dict], pair_type: str) -> Dict:
        """创建单个DPO训练样本（1 chosen + N rejected）"""

        # 计算性能指标
        chosen_acc = chosen['accuracy']
        chosen_mem = chosen['memory_mb']
        chosen_quant = chosen['quantization_mode']
        
        # 对于每个rejected，我们会创建一个独立的pair
        # 这里先准备第一个rejected的信息（稍后会展开）
        rejected_acc = rejected[0]['accuracy']
        rejected_mem = rejected[0]['memory_mb']
        rejected_quant = rejected[0]['quantization_mode']

        # 🎲 随机化性能描述顺序
        if random.random() < 0.5:
            # Chosen的性能在前
            perf_hint = f"One achieves {chosen_acc:.2f}% accuracy with {chosen_mem:.2f}MB memory using {chosen_quant} quantization, and the other achieves {rejected_acc:.2f}% accuracy with {rejected_mem:.2f}MB memory using {rejected_quant} quantization."
        else:
            # Rejected的性能在前
            perf_hint = f"One achieves {rejected_acc:.2f}% accuracy with {rejected_mem:.2f}MB memory using {rejected_quant} quantization, and the other achieves {chosen_acc:.2f}% accuracy with {chosen_mem:.2f}MB memory using {chosen_quant} quantization."

        # Instruction （固定）
        instruction = f"""You are a neural network architecture design expert. Your task is to generate improved network configurations based on given constraints.


**Conv Type in the Search Space:**
    1. DWSepConvBlock: Depthwise separable convolution (Depthwise + Pointwise) structure with skip connection support.
    2. MBConvBlock: Inverted residual structure (expansion convolution + Depthwise + SE module + Pointwise) with skip connection support.
    3. DpConvBlock: Pure depthwise convolution (Depthwise + Pointwise) structure without SE module or skip connections.
    4. SeSepConvBlock: Depthwise separable convolution with SE module (Depthwise + SE + Pointwise) structure.
    5. SeDpConvBlock: Depthwise convolution with SE module (Depthwise + SE) structure without Pointwise convolution.

**CRITICAL: Memory Consumption Ranking (Single Stage, Single Block Comparison):**

        LIGHTEST → HEAVIEST (Mhealth):
        1. SeDpConv:   (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
        2. DpConv:   (Depthwise + Pointwise, no SE)
        3. DWSepConv:  (Depthwise + Pointwise + optional skip)
        4. SeSepConv:  (Depthwise + SE + Pointwise)
        5. MBConv:   (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
        

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

**Task Context:**
- Dataset: Mhealth
  - Input Channels: 6
  - Time Steps: 300
  - Number of Classes: 27
  - Description: The chest-and-wrist HAR dataset includes standing, sitting, supine, and lateral lying postures (and so on).
- Memory Constraint: ≤ {self.memory_limit} MB
- Optimization Goal: Maximize accuracy while satisfying memory constraint

**Search Space Constraints:**
- Stages: 1-4
- Conv types: DWSepConv, MBConv, DpConv, SeSepConv, SeDpConv
- Kernel sizes: 3, 5, 7
- Strides: 1, 2, 4
- Channels: 6, 12, 18, 24
- Blocks per stage: 1-2
- Quantization modes: none, static, qat, qaft
- Skip connection: True, False
- Activations: ReLU6, LeakyReLU, Swish
- Expansions: 1, 2, 3, 4
- SE ratios: 0, 0.25, 0.5

**EVALUATION TASK:**
You are given two architecture configurations below. {perf_hint}

**Your task is to:**
1. Analyze both configurations to understand their architectural differences
2. Determine which configuration corresponds to which performance metrics
3. Select the BETTER configuration based on these criteria (in priority order):
   a) MUST satisfy memory constraint (≤ {self.memory_limit} MB) - configurations exceeding this are INVALID
   b) Among valid configurations, prefer higher accuracy

**Think step by step:**
- First, estimate each configuration's memory footprint based on the architecture design
- Check if each configuration satisfies the memory constraint with different quantization modes
- Quantization Impact: static/qat/qaft modes reduce memory to 1/4 of FP32 size
- Compare accuracy of valid configurations
- Select the configuration that best meets the criteria

Respond with ONLY the JSON configuration for the architecture, no explanation or additional text."""
        
        # Chosen output
        chosen_config_json = json.dumps(chosen['config'], indent=2)
        # 计算chosen的性能指标
        chosen_reward = self.compute_reward(chosen)
        chosen_utilization = chosen['memory_mb'] / self.memory_limit * 100

        chosen_output = f"""{chosen_config_json}"""
#         **Performance Metrics:**
# - Accuracy: {chosen['accuracy']:.2f}%
# - Memory: {chosen['memory_mb']:.2f} MB / {self.memory_limit} MB ({chosen_utilization:.1f}% utilized)
# - Quantization: {chosen['quantization_mode']}
# - Status: ✅ VALID (meets memory constraint)
# - Evaluation Score: {chosen_reward:.2f}

        rejected_outputs = []
        for r in rejected:
            r_reward = self.compute_reward(r)
            r_utilization = r['memory_mb'] / self.memory_limit * 100
            
            rejected_config_json = json.dumps(r['config'], indent=2)
            
            # 判断被拒绝的原因
            if r['memory_mb'] > self.memory_limit:
                overshoot = r['memory_mb'] - self.memory_limit
                status = f"❌ INVALID"
                reason = f"Memory constraint violation: Exceeds limit by {overshoot:.2f} MB"
                detail = f"Cannot be deployed: Memory requirement ({r['memory_mb']:.2f}MB) > Available memory ({self.memory_limit}MB)"
            elif r['accuracy'] < chosen['accuracy'] - 5:
                gap = chosen['accuracy'] - r['accuracy']
                status = f"⚠️ SUBOPTIMAL"
                reason = f"Significantly lower accuracy: {gap:.2f}% below optimal"
                detail = "Valid but poor performance"
            elif r['accuracy'] < chosen['accuracy']:
                gap = chosen['accuracy'] - r['accuracy']
                status = f"⚠️ SUBOPTIMAL"
                reason = f"Lower accuracy: {gap:.2f}% below optimal"
                detail = "Valid but suboptimal performance"
            else:
                # rejected准确率>=chosen（ 来自violation pairs ）
                status = f"❌ INVALID"
                reason = "Memory constraint violation"
                detail = f"Despite higher accuracy ({r['accuracy']:.2f}% vs {chosen['accuracy']:.2f}%), cannot be used due to memory constraint"
            
            rejected_output = f"""{rejected_config_json}"""
            rejected_outputs.append(rejected_output)  
            
# **Performance Metrics:**
# - Accuracy: {r['accuracy']:.2f}%
# - Memory: {r['memory_mb']:.2f} MB / {self.memory_limit} MB ({r_utilization:.1f}% utilized)
# - Quantization: {r['quantization_mode']}
# - Status: {status}
# - Reason: {reason}
# - Detail: {detail}
# - Evaluation Score: {r_reward:.2f}

# **Why Rejected:**
# - Chosen Score: {chosen_reward:.2f} (Valid + Accurate)
# - This Config Score: {r_reward:.2f} ({'Invalid' if r_reward == 0 else 'Valid but Lower Accuracy'})
# - Decision: Chosen is {'usable' if chosen_reward > 0 else 'N/A'}, This config is {'NOT usable' if r_reward == 0 else 'usable but worse'}
                          
        
        return {
            'prompt': instruction,
            'chosen': chosen_output,
            'rejected': rejected_outputs,  # 列表
            'metadata': {
                'type': pair_type,
                'chosen_accuracy': chosen['accuracy'],
                'chosen_memory': chosen['memory_mb'],
                'chosen_reward': chosen_reward,  # ← 添加这个！
                'chosen_quant': chosen['quantization_mode'],
                'chosen_node_id': chosen['node_id'],
                'rejected_info': [
                    {
                        'accuracy': r['accuracy'],
                        'memory': r['memory_mb'],
                        'reward': self.compute_reward(r),  # ← 添加这个！
                        'quant': r['quantization_mode'],
                        'node_id': r['node_id']
                    }
                    for r in rejected
                ]
            }
        }
    
    def deduplicate_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """去重（基于chosen+rejected的组合）"""
        seen = set()
        unique_pairs = []
        
        for pair in pairs:
            # 创建唯一key
            rejected_str = '|'.join(sorted(pair['rejected']))
            key = hash(pair['chosen'] + rejected_str)
            
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)
        
        return unique_pairs
    
    def save_dataset(self, pairs: List[Dict], output_path: str):
        """保存为DPO训练格式"""
        
        # 展开：每个 rejected 创建一个独立的训练样本
        expanded_pairs = []
        for pair in pairs:
            for rejected_output in pair['rejected']:
                expanded_pairs.append({
                    'prompt': pair['prompt'],
                    'chosen': pair['chosen'],
                    'rejected': rejected_output
                })
        
        print(f"\n💾 Saving dataset...")
        print(f"  Original pairs: {len(pairs)}")
        print(f"  Expanded pairs (1 chosen + 1 rejected each): {len(expanded_pairs)}")
        
        # 保存训练用（简化版）
        with open(output_path, 'w') as f:
            for pair in expanded_pairs:
                f.write(json.dumps({
                    'prompt': pair['prompt'],
                    'chosen': pair['chosen'],
                    'rejected': pair['rejected']
                }) + '\n')
        
        print(f"  ✅ Saved to {output_path}")
        
        # 保存debug用（带metadata）
        debug_path = output_path.replace('.jsonl', '_debug.jsonl')
        with open(debug_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair, indent=2) + '\n')
        
        print(f"  ✅ Saved debug version to {debug_path}")
    
    def print_statistics(self, pairs: List[Dict]):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 DPO Dataset Statistics")
        print("="*60)
        
        # 计算展开后的数量
        total_expanded = sum(len(p['rejected']) for p in pairs)
        
        print(f"\nDataset size:")
        print(f"  Unique (chosen, rejected_list) groups: {len(pairs)}")
        print(f"  Total training pairs (expanded): {total_expanded}")
        print(f"  Average rejected per chosen: {total_expanded / len(pairs):.1f}")
        
        # 按类型统计
        by_type = defaultdict(int)
        for pair in pairs:
            by_type[pair['metadata']['type']] += 1
        
        print(f"\nPairs by type:")
        for ptype, count in sorted(by_type.items()):
            print(f"  {ptype:15s}: {count:5d} ({count/len(pairs)*100:5.1f}%)")
        
        # Chosen准确率统计
        chosen_accs = [p['metadata']['chosen_accuracy'] for p in pairs]
        print(f"\nChosen accuracy distribution:")
        print(f"  Min:    {min(chosen_accs):.2f}%")
        print(f"  Max:    {max(chosen_accs):.2f}%")
        print(f"  Mean:   {sum(chosen_accs)/len(chosen_accs):.2f}%")
        print(f"  Median: {sorted(chosen_accs)[len(chosen_accs)//2]:.2f}%")
        
        # Accuracy gap统计
        acc_gaps = []
        for pair in pairs:
            chosen_acc = pair['metadata']['chosen_accuracy']
            for r_info in pair['metadata']['rejected_info']:
                gap = chosen_acc - r_info['accuracy']
                acc_gaps.append(gap)
        
        if acc_gaps:
            print(f"\nAccuracy gap (chosen - rejected):")
            print(f"  Mean:   {sum(acc_gaps)/len(acc_gaps):6.2f}%")
            print(f"  Median: {sorted(acc_gaps)[len(acc_gaps)//2]:6.2f}%")
            print(f"  Min:    {min(acc_gaps):6.2f}%")
            print(f"  Max:    {max(acc_gaps):6.2f}%")

        # 新增：Reward gap统计
        reward_gaps = []
        for pair in pairs:
            chosen_reward = pair['metadata'].get('chosen_reward', 0)
            for r_info in pair['metadata']['rejected_info']:
                r_reward = r_info.get('reward', 0)
                gap = chosen_reward - r_reward
                reward_gaps.append(gap)
        
        if reward_gaps:
            print(f"\n🎯 Reward gap (chosen - rejected):")
            print(f"  Mean:   {sum(reward_gaps)/len(reward_gaps):6.2f}")
            print(f"  Median: {sorted(reward_gaps)[len(reward_gaps)//2]:6.2f}")
            print(f"  Min:    {min(reward_gaps):6.2f}")
            print(f"  Max:    {max(reward_gaps):6.2f}")
        
        # 显示样本（增强版）
        print("\n" + "="*60)
        print("📝 Sample Pairs")
        print("="*60)
        
        for i, sample in enumerate(random.sample(pairs, min(3, len(pairs))), 1):
            print(f"\n--- Sample {i} ({sample['metadata']['type']}) ---")
            
            chosen_reward = sample['metadata'].get('chosen_reward', 0)
            print(f"Chosen:")
            print(f"  Accuracy: {sample['metadata']['chosen_accuracy']:.2f}%")
            print(f"  Memory: {sample['metadata']['chosen_memory']:.2f}MB")
            print(f"  Reward: {chosen_reward:.2f}")
            print(f"  Quant: {sample['metadata']['chosen_quant']}")
            
            for j, r_info in enumerate(sample['metadata']['rejected_info'], 1):
                r_reward = r_info.get('reward', 0)
                print(f"Rejected {j}:")
                print(f"  Accuracy: {r_info['accuracy']:.2f}%")
                print(f"  Memory: {r_info['memory']:.2f}MB")
                print(f"  Reward: {r_reward:.2f}")
                print(f"  Quant: {r_info['quant']}")
                print(f"  Gap: Acc {sample['metadata']['chosen_accuracy'] - r_info['accuracy']:+.2f}%, "
                    f"Reward {chosen_reward - r_reward:+.2f}")


# ============ 使用示例 ============
if __name__ == "__main__":
    # 你的数据文件路径
    # Mhealth
        # "/root/unified_nas/results/sfs_har_Mhealth_20251205_141342/search_state.json",
        # "/root/unified_nas/results/sfs_har_Mhealth_20251205_141046/search_state.json",
        # "/root/unified_nas/results/sfs_har_Mhealth_20251210_130329/search_state.json",
        # "/root/unified_nas/results/sfs_har_Mhealth_20251210_130710/search_state.json"
    # MMAct 
        # "/root/unified_nas/results/sfs_har_MMAct_20251218_022441/search_state.json",
        # "/root/unified_nas/results/sfs_har_MMAct_20251218_022716/search_state.json",
        # "/root/unified_nas/results/sfs_har_MMAct_20251218_023433/search_state.json",
        # "/root/unified_nas/results/sfs_har_MMAct_20260106_131043/search_state.json"
    # Wharf
    #     "/root/unified_nas/results/sfs_har_Wharf_20251227_161545/search_state.json",
    #     "/root/unified_nas/results/sfs_har_Wharf_20251227_161641/search_state.json",
    #     "/root/unified_nas/results/sfs_har_Wharf_20251227_161655/search_state.json",
    #     "/root/unified_nas/results/sfs_har_Wharf_20251227_161738/search_state.json",
    #     "/root/unified_nas/results/sfs_har_Wharf_20260112_072938/search_state.json"

    # USCHAD
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142123/search_state.json  12
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142152/search_state.json  15
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142230/search_state.json  25
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142253/search_state.json  20
    # /root/unified_nas/results/sfs_har_USCHAD_20260112_042628/search_state.json  15

    # UTD-MHAD
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135544/search_state.json 5MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135632/search_state.json 10MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084243/search_state.json 8MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084245/search_state.json 3MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/search_state.json 3MB


    search_state_files = [
        "/root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135544/search_state.json",
        "/root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135632/search_state.json",
        "/root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084243/search_state.json",
        "/root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084245/search_state.json",
        "/root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/search_state.json"
    ]
    
    # 构建DPO数据集
    builder = MCTSDPOBuilder(
        search_state_paths=search_state_files,
        memory_limit=8.0,
        min_accuracy_threshold=70.0,  # rejected 可以低至 70%
        chosen_accuracy_threshold=73.0  # 新增： chosen 必须 ≥94%
    )
    
    # 生成数据集
    dataset = builder.build_dataset(
        output_path='/root/llama_fine/data_preprocess/dpo_corpus/UTD-MHAD/dpo_8mb.jsonl',
        pairs_per_chosen=2,
        max_pairs=None
    )
    
    print("\n✅ Done! Dataset ready for DPO training.")
    print(f"\n📂 Output files:")
    print(f"  - dpo_mcts_20mb.jsonl (for training)")
    print(f"  - dpo_mcts_20mb_debug.jsonl (for analysis)")