# import json
# from collections import defaultdict

# def analyze_dataset(filepath):
#     stats = defaultdict(int)
#     with open(filepath) as f:
#         for line in f:
#             data = json.loads(line)
#             # 提取内存限制（从prompt中）
#             # 或者从debug文件读取metadata
#             stats['total'] += 1
#     return stats

# for mb in [3, 5, 8]:
#     stats = analyze_dataset(f'/root/llama_fine/data_preprocess/dpo_corpus/MMAct/dpo_{mb}mb.jsonl')
#     print(f"{mb}MB: {stats}")

# 上面是在分析 数据集 发布情况

import json
import random

def merge_datasets_simple(files, output):
    """简单合并+打乱"""
    merged = []
    
    for filepath in files:
        with open(filepath) as f:
            data = [json.loads(line) for line in f]
            merged.extend(data)
            print(f"Loaded {len(data)} pairs from {filepath}")
    
    # 打乱顺序
    random.shuffle(merged)
    
    # 保存
    with open(output, 'w') as f:
        for item in merged:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Merged {len(merged)} pairs to {output}")
    return merged

# 使用
        # '/root/llama_fine/data_preprocess/dpo_corpus/Mhealth/dpo_12mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/Mhealth/dpo_15mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/Mhealth/dpo_20mb.jsonl'

        #         '/root/llama_fine/data_preprocess/dpo_corpus/MMAct1/dpo_3mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/MMAct1/dpo_5mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/MMAct1/dpo_8mb.jsonl'

        #    '/root/llama_fine/data_preprocess/dpo_corpus/Wharf1/dpo_3mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/Wharf1/dpo_5mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/Wharf1/dpo_8mb.jsonl'

        #         '/root/llama_fine/data_preprocess/dpo_corpus/USCHAD/dpo_12mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/USCHAD/dpo_15mb.jsonl',
        # '/root/llama_fine/data_preprocess/dpo_corpus/USCHAD/dpo_20mb.jsonl'

merged = merge_datasets_simple(
    [
        '/root/llama_fine/data_preprocess/dpo_corpus/UTD-MHAD/dpo_3mb_x4.jsonl',
        '/root/llama_fine/data_preprocess/dpo_corpus/UTD-MHAD/dpo_5mb.jsonl',
        '/root/llama_fine/data_preprocess/dpo_corpus/UTD-MHAD/dpo_8mb.jsonl'
    ],
    '/root/llama_fine/data_preprocess/merged/UTD-MHAD/dpo_merged.jsonl'
)