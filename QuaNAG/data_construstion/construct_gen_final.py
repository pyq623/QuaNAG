#!/usr/bin/env python3
"""
从 search_state.json 的 training_trajectories 中提取高性能模型架构
按照 rebuild_data.py 的格式构建增强训练数据集
"""

import json
from typing import Dict, List
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data import get_dataset_info

def extract_and_rebuild_from_search_state(
    dataset: str,
    search_state_file: str,
    output_file: str,
    min_accuracy: float = 90.0,
    top_percentile: float = 0.95,
    repeat_factor: int = 3,
    max_memory: float = 15.0
):
    """
    从 search_state.json 中提取高性能样本并重构为训练数据集
    
    参数:
        search_state_file: search_state.json 文件路径
        output_file: 输出JSONL文件路径
        min_accuracy: 最小准确率阈值（默认90%）
        top_percentile: 前百分比样本进行重复（默认0.95，即前95%）
        repeat_factor: 重复增强倍数（默认3倍）
    """
    
    # 固定的 System Prompt (与 rebuild_data.py 完全一致)
    SYSTEM_PROMPT = """You are a neural network architecture design expert. Your task is to generate improved network configurations based on given constraints and insights.


**Conv Type in the Search Space:**
    1. DWSepConvBlock: Depthwise separable convolution (Depthwise + Pointwise) structure with skip connection support.
    2. MBConvBlock: Inverted residual structure (expansion convolution + Depthwise + SE module + Pointwise) with skip connection support.
    3. DpConvBlock: Pure depthwise convolution (Depthwise + Pointwise) structure without SE module or skip connections.
    4. SeSepConvBlock: Depthwise separable convolution with SE module (Depthwise + SE + Pointwise) structure.
    5. SeDpConvBlock: Depthwise convolution with SE module (Depthwise + SE) structure without Pointwise convolution.

**CRITICAL: Memory Consumption Ranking (Single Stage, Single Block Comparison):**

        LIGHTEST → HEAVIEST (Mhealth):
        1. SeDpConv  (Only Depthwise + SE, NO Pointwise - MOST LIGHTWEIGHT)
        2. DpConv   (Depthwise + Pointwise, no SE)
        3. DWSepConv  (Depthwise + Pointwise + optional skip)
        4. SeSepConv  (Depthwise + SE + Pointwise)
        5. MBConv    (Expansion + Depthwise + SE + Pointwise - MOST HEAVY)
        

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
    
Please ensure the configurations adhere to the defined search space constraints."""
    
    # 基础 Task Prompt 模板（max_memory将动态填充）
    def get_base_task_prompt(max_memory: float, dataset: str) -> str:
        dataset_info = get_dataset_info(dataset)
        return f"""
**Task Context:**
- Dataset: {dataset}
  - Input Channels: {dataset_info.get('channels', 'N/A')}
  - Time Steps: {dataset_info.get('time_steps', 'N/A')}
  - Number of Classes: {dataset_info.get('num_classes', 'N/A')}
  - Description: The chest-and-wrist HAR dataset includes standing, sitting, supine, and lateral lying postures (and so on).
- Memory Constraint: ≤ {max_memory:.1f} MB
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
- SE ratios: 0, 0.25, 0.5"""
    
    print("🚀 开始从 search_state.json 提取高性能样本...")
    print(f"输入文件: {search_state_file}")
    print(f"输出文件: {output_file}\n")
    
    # 加载 search_state.json
    print("📖 读取 search_state.json...")
    with open(search_state_file, 'r', encoding='utf-8') as f:
        search_state = json.load(f)
    
    training_trajectories = search_state.get('training_trajectories', [])
    
    print(f"✓ 找到 {len(training_trajectories)} 条训练轨迹\n")
    
    # 统计信息
    stats = {
        'total_trajectories': len(training_trajectories),
        'high_quality': 0,
        'samples_before_repeat': 0,
        'samples_after_repeat': 0,
        'skipped_low_accuracy': 0,
        'top_percentile_count': 0,
        'accuracy_distribution': {
            '72-100': 0,      # ← 修改这里
            '69-72': 0,       # ← 修改这里
            '68-69': 0,       # ← 修改这里
            '65-68': 0,       # ← 修改这里
            '<65': 0          # ← 修改这里
        }
    }
    
    # 第一遍：提取所有高质量样本
    all_samples = []
    
    print(f"🔍 提取高质量样本 (accuracy >= {min_accuracy}%)...")
    
    for idx, traj in enumerate(training_trajectories):
        trajectory_id = traj.get('trajectory_id')
        child = traj.get('child', {})
        
        # 提取子节点的配置和准确率
        child_config = child.get('config')
        child_metrics = child.get('metrics', {})
        accuracy = child_metrics.get('accuracy')
        memory_mb = child_metrics.get('memory_mb')
        node_id = child.get('node_id')
        
        if not child_config or accuracy is None:
            print(f"  ⚠️  轨迹 {idx}: 缺少配置或准确率，跳过")
            continue
        
        # 统计准确率分布
        if accuracy >= 72:
            stats['accuracy_distribution']['72-100'] += 1
        elif accuracy >= 69:
            stats['accuracy_distribution']['69-72'] += 1
        elif accuracy >= 68:
            stats['accuracy_distribution']['68-69'] += 1
        elif accuracy >= 65:
            stats['accuracy_distribution']['65-68'] += 1
        else:
            stats['accuracy_distribution']['<65'] += 1
        
        # 过滤低准确率样本
        if accuracy < min_accuracy:
            stats['skipped_low_accuracy'] += 1
            continue
        
        stats['high_quality'] += 1
        
        # 获取 max_memory（默认15.0 MB）
        max_memory = max_memory  # 根据你的搜索空间设定
        dataset_info = get_dataset_info(dataset)
        # 确保config中有必要的字段
        if 'input_channels' not in child_config:
            child_config['input_channels'] = dataset_info.get('channels', 9)
        if 'num_classes' not in child_config:
            child_config['num_classes'] = dataset_info.get('num_classes', 6)
        
        # 构造包含Expected Performance的完整Instruction
        performance_section = f"""

**Expected Architecture Performance (for reference):**
- Target Accuracy: ~{accuracy:.2f}%
- Target Memory: ~{memory_mb:.2f} MB

**Task:**
Generate an architecture configuration that:
1. Achieves the target accuracy (~{accuracy:.2f}%)
2. Approaches the target memory (~{memory_mb:.2f} MB) while satisfying the constraint (≤ {max_memory:.1f} MB)
3. Follows all search space rules
4. Uses appropriate architectural choices for the performance target

Respond with ONLY the JSON configuration for the architecture, no explanation or additional text."""
        
        # 完整的instruction = System Prompt + Base Task (with max_memory) + Performance + Final Task
        base_task_prompt = get_base_task_prompt(max_memory, dataset)
        full_instruction = SYSTEM_PROMPT + "\n\n" + base_task_prompt + performance_section
        
        # Output只包含纯配置（不包含accuracy和memory）
        output_config = json.dumps(child_config, indent=2, ensure_ascii=False)
        
        # 构造新样本
        new_sample = {
            'instruction': full_instruction,
            'input': '',
            'output': output_config,
            '_metadata': {
                'node_id': node_id,
                'trajectory_id': trajectory_id,
                'original_index': idx,
                'accuracy': accuracy,
                'memory_mb': memory_mb,
                'max_memory': max_memory,
                'quantization_mode': child_metrics.get('quantization_mode', 'unknown')
            }
        }
        
        all_samples.append(new_sample)
    
    stats['samples_before_repeat'] = len(all_samples)
    
    print(f"\n✓ 提取完成，获得 {len(all_samples)} 个高质量样本\n")
    
    if not all_samples:
        print("❌ 没有找到符合条件的样本！")
        return [], stats
    
    # 按准确率降序排序
    print("📊 按准确率排序...")
    all_samples.sort(key=lambda x: x['_metadata']['accuracy'], reverse=True)
    
    # 计算前 top_percentile 的样本数量
    top_k = int(len(all_samples) * top_percentile)
    stats['top_percentile_count'] = top_k
    
    print(f"✓ 前 {int(top_percentile*100)}% 样本数: {top_k}\n")
    
    # 第二遍：对前95%的样本进行重复增强
    print(f"🔄 数据增强 (前{int(top_percentile*100)}%样本重复{repeat_factor}倍)...")
    enhanced_samples = []
    
    for idx, sample in enumerate(all_samples):
        if idx < top_k:
            # 前95%的样本重复repeat_factor次
            for _ in range(repeat_factor):
                enhanced_samples.append(sample)
        else:
            # 剩余5%的样本保留一次
            enhanced_samples.append(sample)
    
    stats['samples_after_repeat'] = len(enhanced_samples)
    
    print(f"✓ 增强完成，总样本数: {len(enhanced_samples)}\n")
    
    # 保存到JSONL文件
    print(f"💾 保存到文件: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in enhanced_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ 保存完成\n")
    
    # 打印统计信息
    print("="*80)
    print("📊 数据集统计")
    print("="*80)
    print(f"原始轨迹总数:          {stats['total_trajectories']}")
    print(f"✓ 高质量样本:          {stats['high_quality']} (accuracy >= {min_accuracy}%)")
    print(f"✗ 跳过 - 低准确率:     {stats['skipped_low_accuracy']}")
    
    print(f"\n准确率分布 (原始数据):")
    for range_str, count in stats['accuracy_distribution'].items():
        percentage = (count / stats['total_trajectories'] * 100) if stats['total_trajectories'] > 0 else 0
        print(f"  {range_str:>10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n数据增强策略:")
    print(f"  - 最小准确率阈值:    {min_accuracy}%")
    print(f"  - 前 {int(top_percentile*100)}% 样本重复:  {stats['top_percentile_count']} 个")
    print(f"  - 重复增强倍数:      {repeat_factor}x")
    
    print(f"\n样本数量变化:")
    print(f"  - 增强前:            {stats['samples_before_repeat']}")
    print(f"  - 增强后:            {stats['samples_after_repeat']}")
    if stats['samples_before_repeat'] > 0:
        print(f"  - 增长倍数:          {stats['samples_after_repeat'] / stats['samples_before_repeat']:.2f}x")
    
    print(f"\n输出文件: {output_file}")
    print("="*80)
    
    # 打印准确率统计
    if all_samples:
        accuracies = [s['_metadata']['accuracy'] for s in all_samples]
        print(f"\n📈 准确率统计 (>={min_accuracy}% 样本):")
        print(f"  - 最高: {max(accuracies):.2f}%")
        print(f"  - 最低: {min(accuracies):.2f}%")
        print(f"  - 平均: {sum(accuracies)/len(accuracies):.2f}%")
        print(f"  - 中位数: {sorted(accuracies)[len(accuracies)//2]:.2f}%")
        if top_k > 0 and top_k <= len(all_samples):
            print(f"  - 前{int(top_percentile*100)}%分界点: {all_samples[top_k-1]['_metadata']['accuracy']:.2f}%")
        print("="*80)
    
    # 打印示例样本
    if enhanced_samples:
        print(f"\n📝 示例样本 (准确率最高的样本):")
        print("="*80)
        first = enhanced_samples[0]
        
        # 提取instruction中的性能目标
        inst = first['instruction']
        perf_start = inst.find('**Expected Architecture Performance')
        perf_section = inst[perf_start:perf_start+300] if perf_start != -1 else ''
        
        print(f"Instruction包含:")
        print(f"  - System Prompt (架构设计专家提示)")
        print(f"  - Task Context (数据集、约束等)")
        print(f"  - Expected Performance (性能目标) ⭐")
        if perf_section:
            print(f"\n性能目标部分预览:")
            for line in perf_section.split('\n')[:5]:
                if line.strip():
                    print(f"    {line}")
        
        print(f"\nInput: '{first['input']}'")
        print(f"\nOutput (纯配置，前500字符):")
        print(first['output'][:500])
        if len(first['output']) > 500:
            print("...")
        
        if '_metadata' in first:
            meta = first['_metadata']
            print(f"\n元数据 (仅供参考，不在训练数据中):")
            print(f"  - node_id:          {meta.get('node_id', 'N/A')}")
            print(f"  - trajectory_id:    {meta.get('trajectory_id', 'N/A')}")
            print(f"  - original_index:   {meta['original_index']}")
            print(f"  - accuracy:         {meta['accuracy']:.2f}% ⭐")
            print(f"  - memory_mb:        {meta['memory_mb']:.2f} MB" if meta.get('memory_mb') else "  - memory_mb:        N/A")
            print(f"  - quantization:     {meta['quantization_mode']}")
        
        print("\n✨ 关键特征:")
        print("  ✓ Instruction中包含Target Accuracy和Target Memory")
        print("  ✓ Output只包含纯配置（不含accuracy）")
        print("  ✓ 模型学习: 性能目标 → 架构设计")
        print("="*80)
    
    return enhanced_samples, stats


def verify_dataset(dataset_file: str):
    """验证生成的数据集"""
    print("\n" + "="*80)
    print("🔍 验证数据集...")
    print("="*80)
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总行数: {len(lines)}")
    
    if not lines:
        print("❌ 数据集为空！")
        return False
    
    # 验证第一个样本
    try:
        first_sample = json.loads(lines[0])
        
        # 检查必需字段
        assert 'instruction' in first_sample, "缺少 instruction 字段"
        assert 'input' in first_sample, "缺少 input 字段"
        assert 'output' in first_sample, "缺少 output 字段"
        
        print("✓ 基础字段验证通过")
        
        # 验证instruction中包含Expected Performance
        assert 'Expected Architecture Performance' in first_sample['instruction'], \
            "Instruction中缺少Expected Performance部分"
        assert 'Target Accuracy' in first_sample['instruction'], \
            "Instruction中缺少Target Accuracy"
        assert 'Target Memory' in first_sample['instruction'], \
            "Instruction中缺少Target Memory"
        
        print("✓ Instruction格式验证通过（包含Expected Performance）")
        
        # 验证output是纯配置（不包含accuracy）
        config = json.loads(first_sample['output'])
        
        # 确保output是config，不包含expected_accuracy
        assert 'input_channels' in config, "配置缺少 input_channels"
        assert 'num_classes' in config, "配置缺少 num_classes"
        assert 'quant_mode' in config, "配置缺少 quant_mode"
        assert 'stages' in config, "配置缺少 stages"
        
        # 确保output不包含expected_accuracy（应该只在instruction中）
        assert 'expected_accuracy' not in config, \
            "Output不应包含expected_accuracy（应该只在instruction中）"
        
        print("✓ Output格式验证通过（纯配置，不含accuracy）")
        
        # 验证元数据
        if '_metadata' in first_sample:
            meta = first_sample['_metadata']
            assert 'accuracy' in meta, "元数据缺少 accuracy"
            print("✓ 元数据验证通过")
        
        print(f"\n验证详情:")
        print(f"  - Instruction包含Target Accuracy: ✓")
        print(f"  - Instruction包含Target Memory: ✓")
        print(f"  - Output是纯配置（无accuracy）: ✓")
        print(f"  - Config包含 {len(config['stages'])} 个stages")
        print(f"  - Quantization mode: {config['quant_mode']}")
        
        # 提取并显示instruction中的性能目标
        inst = first_sample['instruction']
        if 'Target Accuracy: ~' in inst:
            acc_line = [line for line in inst.split('\n') if 'Target Accuracy:' in line][0]
            mem_line = [line for line in inst.split('\n') if 'Target Memory:' in line][0]
            print(f"\nInstruction中的性能目标:")
            print(f"  {acc_line.strip()}")
            print(f"  {mem_line.strip()}")
        
        print("\n✅ 数据集验证通过！")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 配置文件路径
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142123/search_state.json  12
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142152/search_state.json  15
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142230/search_state.json  25
    # /root/unified_nas/results/sfs_har_USCHAD_20251223_142253/search_state.json  20
    # /root/unified_nas/results/sfs_har_USCHAD_20260112_042628/search_state.json  15

#     /root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135544/search_state.json 5MB
# /root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135632/search_state.json 10MB
# /root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084243/search_state.json 8MB
# /root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084245/search_state.json 3MB
# /root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/search_state.json 3MB

    search_state_file = '/root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084245/search_state.json'
    output_file = '/root/llama_fine/data_preprocess/fin_gen_task/UTD-MHAD/nas_generation_3MB.jsonl'
    
    # 提取并重构数据集
    samples, stats = extract_and_rebuild_from_search_state(
        dataset='UTD-MHAD',
        search_state_file=search_state_file,
        output_file=output_file,
        min_accuracy=69.0,      # 只使用 准确率 >= 90% 的样本
        top_percentile=0.72,    # 前 95% 样本进行重复
        repeat_factor=5,         # 重复 4 倍
        max_memory=3.0
    )
    
    print("\n" + "="*80)
    print("✅ 数据集重构完成！")
    print("="*80)
    
    # 验证生成的数据集
    if samples:
        verify_dataset(output_file)