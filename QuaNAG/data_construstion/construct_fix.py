import json
import re

def extract_json_from_message(content):
    """从消息中提取JSON配置"""
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    return None

def parse_architecture_violations(auto_correction_msg):
    """从AUTO-CORRECTION消息中提取架构违规"""
    violations = []
    content = auto_correction_msg["content"]
    
    # 提取Corrections made部分
    corrections_match = re.search(r'\*\*Corrections made:\*\*(.*?)(?:\*\*Result:|$)', content, re.DOTALL)
    if not corrections_match:
        return violations
    
    corrections_text = corrections_match.group(1)
    
    # 按行分割修正项
    for line in corrections_text.split('\n'):
        line = line.strip()
        if not line or not line.startswith('-'):
            continue
        
        # 移除前导的 "- " 或 "-"
        clean_line = line.lstrip('- ').strip()
        
        # ========== 跳过内存相关的修正（这些属于memory_violation） ==========
        if "Memory reduction:" in clean_line or \
           "Stage" in clean_line and "channels:" in clean_line and "→" in clean_line or \
           "Removed block:" in clean_line or \
           "Removed stage" in clean_line:
            continue
        
        # ========== 1. skip_connection不支持 ==========
        if "does not support skip_connection" in clean_line:
            match = re.search(r'(\w+) does not support skip_connection', clean_line)
            block_type = match.group(1) if match else None
            violations.append({
                "type": "architecture_violation",
                "rule": "skip_connection_not_supported",
                "details": clean_line,
                "block_type": block_type
            })
        
        # ========== 2. DWSepConv expansion>1等价于MBConv ==========
        elif "expansion>1 is equivalent to MBConv" in clean_line or \
             ("expansion" in clean_line.lower() and "→" in clean_line and "DWSepConv" in clean_line):
            violations.append({
                "type": "architecture_violation",
                "rule": "dwsepconv_expansion_equivalence",
                "details": clean_line
            })
        
        # ========== 3. SeDpConv通道不匹配 ==========
        elif "SeDpConv" in clean_line and ("channel" in clean_line.lower() or "in_channels" in clean_line or "out_channels" in clean_line):
            violations.append({
                "type": "architecture_violation",
                "rule": "sedpconv_channel_mismatch",
                "details": clean_line
            })
        
        # ========== 4. has_se与se_ratio不一致 ==========
        elif ("has_se" in clean_line and "se_ratio" in clean_line) or \
             ("has_se" in clean_line.lower() and "0" in clean_line) or \
             ("se_ratio" in clean_line.lower() and ("false" in clean_line.lower() or "true" in clean_line.lower())):
            violations.append({
                "type": "architecture_violation",
                "rule": "se_ratio_inconsistency",
                "details": clean_line
            })
        
        # ========== 5. DpConv不支持has_se/se_ratio ==========
        elif "DpConv" in clean_line and ("has_se" in clean_line or "se_ratio" in clean_line):
            violations.append({
                "type": "architecture_violation",
                "rule": "dpconv_se_not_supported",
                "details": clean_line
            })
        
        # ========== 6. DpConv不支持skip_connection ==========
        elif "DpConv" in clean_line and "skip_connection" in clean_line:
            violations.append({
                "type": "architecture_violation",
                "rule": "dpconv_skip_not_supported",
                "details": clean_line
            })
        
        # ========== 7. SeDpConv在第一个stage的通道约束 ==========
        elif "first stage" in clean_line.lower() and "SeDpConv" in clean_line:
            violations.append({
                "type": "architecture_violation",
                "rule": "sedpconv_first_stage_channel_constraint",
                "details": clean_line
            })
        
        # ========== 8. 其他Auto-fixing消息（通用捕获） ==========
        elif "Auto-fixing:" in clean_line or "Auto-fix:" in clean_line:
            violations.append({
                "type": "architecture_violation",
                "rule": "general_auto_fix",
                "details": clean_line
            })
    
    return violations

def parse_memory_violations(auto_correction_or_feedback_msg):
    """
    从AUTO-CORRECTION或MEMORY FEEDBACK消息中提取内存违规
    
    支持两种来源：
    1. AUTO-CORRECTION中的 "Memory reduction: XXMb → target YYMb"
    2. MEMORY FEEDBACK中的 "Previous config memory: XXMb (limit: YYMb)"
    """
    violations = []
    content = auto_correction_or_feedback_msg["content"]
    
    # ========== 情况1: AUTO-CORRECTION中的内存违规 ==========
    # 匹配 "Memory reduction: 35.18MB → target 15.0MB"
    auto_memory_match = re.search(r'Memory reduction: ([\d.]+)MB → target ([\d.]+)MB', content)
    if auto_memory_match:
        actual_memory = float(auto_memory_match.group(1))
        limit_memory = float(auto_memory_match.group(2))
        
        violations.append({
            "type": "memory_violation",
            "source": "auto_correction",
            "actual_memory_mb": actual_memory,
            "limit_memory_mb": limit_memory,
            "over_budget_percentage": int((actual_memory - limit_memory) / limit_memory * 100),
            "details": f"Memory {actual_memory}MB exceeds limit {limit_memory}MB (detected in auto-correction)"
        })
    
    # ========== 情况2: MEMORY FEEDBACK中的内存违规 ==========
    # 匹配 "Previous config memory: 19.02MB (limit: 15.0MB)"
    feedback_memory_match = re.search(r'Previous config memory: ([\d.]+)MB \(limit: ([\d.]+)MB\)', content)
    over_budget_match = re.search(r'Over budget by: (\d+)%', content)
    
    if feedback_memory_match:
        actual_memory = float(feedback_memory_match.group(1))
        limit_memory = float(feedback_memory_match.group(2))
        over_budget_pct = int(over_budget_match.group(1)) if over_budget_match else None
        
        violations.append({
            "type": "memory_violation",
            "source": "memory_feedback",
            "actual_memory_mb": actual_memory,
            "limit_memory_mb": limit_memory,
            "over_budget_percentage": over_budget_pct,
            "details": f"Memory {actual_memory}MB exceeds limit {limit_memory}MB"
        })
    
    return violations

def extract_repair_task(dialogue):
    """提取修复任务
    
    关键逻辑：
    1. 如果有DEGRADATION → input是倒数第2个assistant config
    2. 如果只有AUTO-CORRECTION → input是倒数第1个assistant config
    """
    messages = dialogue["messages"]
    final_config = dialogue.get("final_config")
    
    if not final_config:
        return None
    
    # 找到所有Assistant生成的config
    assistant_configs = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and "```json" in msg["content"]:
            config = extract_json_from_message(msg["content"])
            if config:
                assistant_configs.append({
                    "index": i,
                    "config": config,
                    "message": msg["content"]
                })
    
    if len(assistant_configs) < 1:
        return None
    
    # 检查是否有DEGRADATION
    has_degradation = False
    for msg in messages:
        if "DEGRADATION TRIGGERED" in msg["content"]:
            has_degradation = True
            break
    
    # 根据情况选择input config
    if has_degradation:
        # DEGRADATION: 倒数第2个
        if len(assistant_configs) < 2:
            return None
        input_config_info = assistant_configs[-2]
    else:
        # AUTO-CORRECTION: 倒数第1个
        input_config_info = assistant_configs[-1]
    
    input_config = input_config_info["config"]
    input_config_index = input_config_info["index"]
    
    # 如果input就是final，没有修复
    if json.dumps(input_config, sort_keys=True) == json.dumps(final_config, sort_keys=True):
        return None
    
    # 收集violations（使用validation的解析函数）
    architecture_violations = []
    memory_violations = []
    
    for j in range(input_config_index + 1, len(messages)):
        msg = messages[j]
        
        # 检查AUTO-CORRECTION
        if "AUTO-CORRECTION APPLIED" in msg["content"]:
            arch_viols = parse_architecture_violations(msg)
            architecture_violations.extend(arch_viols)
            
            mem_viols = parse_memory_violations(msg)
            memory_violations.extend(mem_viols)
        
        # 检查MEMORY FEEDBACK
        if ("MEMORY FEEDBACK" in msg["content"] or "Over budget by" in msg["content"]) and "Previous config memory" in msg["content"]:
            mem_viols = parse_memory_violations(msg)
            memory_violations.extend(mem_viols)
    
    # 去重：优先保留MEMORY FEEDBACK的内存违规
    if len(memory_violations) > 1:
        feedback_viols = [v for v in memory_violations if v.get("source") == "memory_feedback"]
        if feedback_viols:
            memory_violations = feedback_viols
        else:
            memory_violations = [memory_violations[0]]
    
    # 如果既没有架构违规也没有内存违规，说明这不是一个修复任务
    if not architecture_violations and not memory_violations:
        return None
    
    all_violations = architecture_violations + memory_violations
    
    repair_task = {
        "dialogue_id": dialogue["dialogue_id"],
        "config": input_config,
        "violations": all_violations,
        "final_config": final_config,
        "has_architecture_violation": len(architecture_violations) > 0,
        "has_memory_violation": len(memory_violations) > 0,
        "has_degradation": has_degradation,
        "total_attempts": dialogue.get("total_attempts", 1)
    }
    
    return repair_task

def process_all_dialogues(dialogues):
    repair_dataset = []
    for dialogue in dialogues:
        repair_task = extract_repair_task(dialogue)
        if repair_task:
            repair_dataset.append(repair_task)
    return repair_dataset

def save_repair_dataset(dataset, output_file):
    """保存为修复数据集格式"""
    repair_tasks = []
    
    # 复用 validation 的 instruction （到 VALIDATION TASK 之前）
    instruction = """ 
    CONSTRAINTS (Search Space):
        Max memory: 3 MB
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

    ## REPAIR TASK
        Given an invalid network configuration with violations, generate a VALID corrected configuration that:
        1. Resolves all architecture violations
        2. Meets memory constraints
        3. Maintains or improves performance
    """
    
    for item in dataset:
        task = {
            "dialogue_id": item["dialogue_id"],
            "instruction": instruction,
            "input": {
                "config": item["config"],
            },
            "output": {
                "violations": item["violations"],
                "fixed_config": item["final_config"]
            },
            # "output": item["final_config"],
            # "violations": item["violations"],
            "metadata": {
                "has_architecture_violation": item["has_architecture_violation"],
                "has_memory_violation": item["has_memory_violation"],
                "has_degradation": item["has_degradation"],
                "total_attempts": item["total_attempts"]
            }
        }
        repair_tasks.append(task)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(repair_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(repair_tasks)} repair tasks to {output_file}")

if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/dialogue_records.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '/root/llama_fine/data_preprocess/fix_task/UTD-MHAD/repair_3MB(1).json'

    with open(input_file, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    dataset = process_all_dialogues(dialogues)
    
    # 统计信息
    total = len(dataset)
    arch_count = sum(1 for d in dataset if d["has_architecture_violation"])
    mem_count = sum(1 for d in dataset if d["has_memory_violation"])
    both_count = sum(1 for d in dataset if d["has_architecture_violation"] and d["has_memory_violation"])
    degradation_count = sum(1 for d in dataset if d.get("has_degradation", False))
    auto_correction_count = total - degradation_count
    
    print(f"Total repair tasks: {total}")
    print(f"  - AUTO-CORRECTION: {auto_correction_count} (last config → final)")
    print(f"  - DEGRADATION: {degradation_count} (second-to-last → final)")
    print(f"\nViolation breakdown:")
    print(f"  - Architecture only: {arch_count - both_count}")
    print(f"  - Memory only: {mem_count - both_count}")
    print(f"  - Both: {both_count}")
    
    # 打印violation类型统计
    from collections import Counter
    all_rules = []
    for item in dataset:
        for v in item["violations"]:
            if v["type"] == "architecture_violation":
                all_rules.append(v["rule"])
    
    if all_rules:
        rule_counts = Counter(all_rules)
        print("\nArchitecture violation types:")
        for rule, count in rule_counts.most_common():
            print(f"  - {rule}: {count}")
    
    save_repair_dataset(dataset, output_file)