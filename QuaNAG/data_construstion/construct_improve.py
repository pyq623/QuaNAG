import json
import sys

def extract_improvement_tasks(search_state):
    """从搜索状态中提取accuracy提升任务"""
    trajectories = search_state.get("training_trajectories", [])
    nodes = search_state.get("nodes", {})
    
    improvement_tasks = []
    
    for traj in trajectories:
        parent = traj.get("parent", {})
        child = traj.get("child", {})
        
        # 获取node_id
        parent_node_id = parent.get("node_id")
        child_node_id = child.get("node_id")
        
        if not parent_node_id or not child_node_id:
            continue
        
        # 从nodes字段中查找对应的metrics
        parent_node = nodes.get(parent_node_id, {})
        child_node = nodes.get(child_node_id, {})
        
        parent_metrics = parent_node.get("metrics", {})
        child_metrics = child_node.get("metrics", {})
        
        parent_accuracy = parent_metrics.get("accuracy")
        child_accuracy = child_metrics.get("accuracy")
        
        # 只保留child accuracy > parent accuracy的情况
        if parent_accuracy is None or child_accuracy is None:
            continue
        
        if child_accuracy <= parent_accuracy:
            continue
        
        # 提取config（从trajectory中获取）
        parent_config = parent.get("config", {})
        child_config = child.get("config", {})
        
        if not parent_config or not child_config:
            continue
        
        parent_memory = parent_metrics.get("memory_mb", 0)
        child_memory = child_metrics.get("memory_mb", 0)
        
        accuracy_improvement = child_accuracy - parent_accuracy
        
        task = {
            "trajectory_id": traj.get("trajectory_id"),
            "parent_node_id": parent_node_id,
            "child_node_id": child_node_id,
            "parent_config": parent_config,
            "parent_metrics": {
                "accuracy": parent_accuracy,
                "memory_mb": parent_memory,
                "quantization_mode": parent_metrics.get("quantization_mode", "none")
            },
            "child_config": child_config,
            "child_metrics": {
                "accuracy": child_accuracy,
                "memory_mb": child_memory,
                "quantization_mode": child_metrics.get("quantization_mode", "none")
            },
            "improvement": {
                "accuracy_gain": round(accuracy_improvement, 2),
                "accuracy_gain_percentage": round(accuracy_improvement / parent_accuracy * 100, 2) if parent_accuracy > 0 else 0
            }
        }
        
        improvement_tasks.append(task)
    
    return improvement_tasks

def save_improvement_dataset(tasks, output_file, memory_limit=20.0):
    """保存为accuracy提升数据集格式"""
    
    instruction_template = """CONSTRAINTS (Search Space):
        Max memory: {memory_limit} MB
        Input: 6 channels, 300 time steps
        Output: 27 classes

    SEARCH SPACE:
        stages: [1, 2, 3, 4]
        conv_types: ["DWSepConv", "MBConv", "DpConv", "SeSepConv", "SeDpConv"]
        kernel_sizes: [3, 5, 7]
        strides: [1, 2, 4]
        skip_connection: [True, False]
        activations: ["ReLU6", "LeakyReLU", "Swish"]
        expansions: [1, 2, 3, 4]
        channels: [6, 12, 18, 24]
        has_se: [True, False]
        se_ratios: [0, 0.25, 0.5]
        blocks_per_stage: [1, 2]
        quantization_modes: ["none", "static", "qat", "qaft"]
    
    **Conv Type in the Search Space:**
        1. DWSepConvBlock: Depthwise separable convolution (Depthwise + Pointwise) structure with skip connection support.
        2. MBConvBlock: Inverted residual structure (expansion convolution + Depthwise + SE module + Pointwise) with skip connection support.
        3. DpConvBlock: Pure depthwise convolution (Depthwise + Pointwise) structure without SE module or skip connections.
        4. SeSepConvBlock: Depthwise separable convolution with SE module (Depthwise + SE + Pointwise) structure.
        5. SeDpConvBlock: Depthwise convolution with SE module (Depthwise + SE) structure without Pointwise convolution.
    **Important Notes:**
        - If has_se is set to False, then se_ratios will be considered as 0, and vice versa. Conversely, if Has_se is set to True, then se_ratios must be greater than 0, and the same holds true in reverse.
        - "MBConv" is only different from "DWSeqConv" when expansion > 1, otherwise they are the same block.
        - If the type of a convolution block is "SeDpConv", then the `in_channels` and `out_channels` of this convolution block must be equal. This means that: - The `out_channels` of the previous convolution block must be equal to both the `in_channels` and `out_channels` of "SeDpConv".
        - If "SeDpConv" is a block in the first stage, its `channels` should be equal to `input_channels`, otherwise an error will be reported.
    **Quantization:**
        • Quantization modes (static/qat/qaft) ALWAYS reduce model memory to EXACTLY 1/4 of FP32 size (INT8 = 8 bits vs FP32 = 32 bits)
        • Example: If FP32 model uses 20MB, quantized version uses 5MB

    ## IMPROVEMENT TASK
        Given a parent configuration with accuracy {parent_accuracy:.2f}% and memory {parent_memory:.2f}MB, generate an IMPROVED configuration that:
        1. Increases accuracy (current: {parent_accuracy:.2f}%)
        2. Stays within memory constraint ({memory_limit}MB)
        3. Follows all architectural rules
"""
    
    dataset = []
    
    for task in tasks:
        parent_acc = task["parent_metrics"]["accuracy"]
        parent_mem = task["parent_metrics"]["memory_mb"]
        child_acc = task["child_metrics"]["accuracy"]
        child_mem = task["child_metrics"]["memory_mb"]
        
        instruction = instruction_template.format(
            memory_limit=memory_limit,
            parent_accuracy=parent_acc,
            parent_memory=parent_mem
        )
        
        item = {
            "trajectory_id": task["trajectory_id"],
            "parent_node_id": task["parent_node_id"],
            "child_node_id": task["child_node_id"],
            "instruction": instruction,
            "input": {
                "config": task["parent_config"],
                "current_metrics": task["parent_metrics"]
            },
            "output": {
                "improved_config": task["child_config"],
                "achieved_metrics": task["child_metrics"]
            },
            "improvement": task["improvement"]
        }
        
        dataset.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(dataset)} improvement tasks to {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/search_state.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '/root/llama_fine/data_preprocess/imporve_task/UTD-MHAD/improvement_3MB(1).json'
    memory_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    
    # 加载搜索状态
    with open(input_file, 'r', encoding='utf-8') as f:
        search_state = json.load(f)
    
    # 提取 improvement 任务
    tasks = extract_improvement_tasks(search_state)
    
    # 统计信息
    total = len(tasks)
    
    if total == 0:
        print("No improvement tasks found (no child accuracy > parent accuracy)")
        sys.exit(0)
    
    # 按accuracy增益排序
    tasks_sorted = sorted(tasks, key=lambda x: x["improvement"]["accuracy_gain"], reverse=True)
    
    print(f"Total improvement tasks: {total}")
    print(f"\nTop 5 improvements:")
    for i, task in enumerate(tasks_sorted[:5]):
        parent_acc = task["parent_metrics"]["accuracy"]
        child_acc = task["child_metrics"]["accuracy"]
        gain = task["improvement"]["accuracy_gain"]
        print(f"  {i+1}. {parent_acc:.2f}% → {child_acc:.2f}% (+{gain:.2f}%)")
    
    # 统计accuracy增益范围
    accuracy_gains = [t["improvement"]["accuracy_gain"] for t in tasks]
    avg_gain = sum(accuracy_gains) / len(accuracy_gains)
    max_gain = max(accuracy_gains)
    min_gain = min(accuracy_gains)
    
    print(f"\nAccuracy gain statistics:")
    print(f"  - Average gain: {avg_gain:.2f}%")
    print(f"  - Max gain: {max_gain:.2f}%")
    print(f"  - Min gain: {min_gain:.2f}%")
    
    # 保存数据集
    save_improvement_dataset(tasks, output_file, memory_limit)