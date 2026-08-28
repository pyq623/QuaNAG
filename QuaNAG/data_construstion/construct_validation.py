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
    """从 AUTO-CORRECTION 消息中提取架构违规"""
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

def extract_first_config_violations(dialogue):
    """提取对话中第一个Assistant config及其所有违规"""
    messages = dialogue["messages"]
    
    # 1. 找到第一个Assistant生成的config
    first_config = None
    first_config_index = -1
    
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and "```json" in msg["content"]:
            first_config = extract_json_from_message(msg["content"])
            first_config_index = i
            break
    
    if not first_config:
        return None
    
    # 2. 收集该config的所有违规
    architecture_violations = []
    memory_violations = []
    
    # 向后查找AUTO-CORRECTION和MEMORY FEEDBACK
    for j in range(first_config_index + 1, len(messages)):
        msg = messages[j]
        
        # 2.1 检查AUTO-CORRECTION
        if "AUTO-CORRECTION APPLIED" in msg["content"]:
            # 提取架构违规
            arch_viols = parse_architecture_violations(msg)
            architecture_violations.extend(arch_viols)
            
            # 提取内存违规（如果AUTO-CORRECTION中包含内存修正）
            mem_viols = parse_memory_violations(msg)
            memory_violations.extend(mem_viols)
        
        # 2.2 检查MEMORY FEEDBACK
        if ("MEMORY FEEDBACK" in msg["content"] or "Over budget by" in msg["content"]) and "Previous config memory" in msg["content"]:
            mem_viols = parse_memory_violations(msg)
            memory_violations.extend(mem_viols)
        
        # 2.3 如果遇到第二个Assistant消息，停止
        if msg["role"] == "assistant" and "```json" in msg["content"] and j != first_config_index:
            break
    
    # 3. 去重：如果同时存在AUTO-CORRECTION和MEMORY FEEDBACK的内存违规，优先保留MEMORY FEEDBACK的
    if len(memory_violations) > 1:
        # 保留source为'memory_feedback'的，如果没有则保留第一个
        feedback_viols = [v for v in memory_violations if v.get("source") == "memory_feedback"]
        if feedback_viols:
            memory_violations = feedback_viols
        else:
            memory_violations = [memory_violations[0]]
    
    # 4. 构造结果
    all_violations = architecture_violations + memory_violations
    
    result = {
        "dialogue_id": dialogue["dialogue_id"],
        "config": first_config,
        "violations": all_violations,
        "has_architecture_violation": len(architecture_violations) > 0,
        "has_memory_violation": len(memory_violations) > 0,
        "is_valid": len(all_violations) == 0
    }
    
    return result

def process_all_dialogues(dialogues):
    """处理所有对话"""
    dataset = []
    
    for dialogue in dialogues:
        result = extract_first_config_violations(dialogue)
        if result:
            dataset.append(result)
    
    return dataset

def save_validation_dataset(dataset, output_file):
    """保存为验证数据集格式"""
    validation_tasks = []
    
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

    ## VALIDATION TASK
        Given a network configuration, identify ALL violations including:
        1. **Architecture violations**: Rule violations in block design
        2. **Memory violations**: Exceeding memory constraint
    """
    
    for item in dataset:
        task = {
            "dialogue_id": item["dialogue_id"],
            "instruction": instruction,
            "input": {
                "config": item["config"],
            },
            "output": {
                "is_valid": item["is_valid"],
                "violations": item["violations"]
            }
        }
        validation_tasks.append(task)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(validation_tasks)} validation tasks to {output_file}")

# 使用示例
if __name__ == "__main__":
    # 加载对话数据
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135544/search_state.json 5MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251226_135632/search_state.json 10MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084243/search_state.json 8MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20251227_084245/search_state.json 3MB
    # /root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/search_state.json 3MB

    with open('/root/unified_nas/results/sfs_har_UTD-MHAD_20260112_072649/dialogue_records.json', 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    # 提取验证数据集
    dataset = process_all_dialogues(dialogues)
    
    # 打印统计信息
    total = len(dataset)
    arch_count = sum(1 for d in dataset if d["has_architecture_violation"])
    mem_count = sum(1 for d in dataset if d["has_memory_violation"])
    valid_count = sum(1 for d in dataset if d["is_valid"])
    
    print(f"Total configs: {total}")
    print(f"  - With architecture violations: {arch_count}")
    print(f"  - With memory violations: {mem_count}")
    print(f"  - Valid configs: {valid_count}")
    
    # 打印violation类型统计
    from collections import Counter
    all_rules = []
    for item in dataset:
        for v in item["violations"]:
            if v["type"] == "architecture_violation":
                all_rules.append(v["rule"])
    
    rule_counts = Counter(all_rules)
    print("\nArchitecture violation types:")
    for rule, count in rule_counts.most_common():
        print(f"  - {rule}: {count}")
    
    # 保存数据集
    save_validation_dataset(dataset, "/root/llama_fine/data_preprocess/validation_task/UTD-MHAD/validation_3MB(1).json")