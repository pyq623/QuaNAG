import json
import random
from pathlib import Path

def load_data(file_path):
    """
    加载 JSON 或 JSONL 文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        list: 数据列表
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.jsonl':
        # JSONL格式：每行一个JSON对象
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    data.append(json.loads(line))
        return data
    else:
        # JSON格式：整个文件是一个JSON数组
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def save_data(data, file_path, format='json'):
    """
    保存数据为JSON或JSONL格式
    
    Args:
        data: 数据列表
        file_path: 输出文件路径
        format: 'json' 或 'jsonl'
    """
    if format == 'jsonl':
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def merge_datasets(input_files, output_file, shuffle=True, output_format='json'):
    """
    合并多个JSON/JSONL数据集文件
    
    Args:
        input_files: 输入文件路径列表
        output_file: 输出文件路径
        shuffle: 是否打乱顺序
        output_format: 输出格式 'json' 或 'jsonl'
    """
    all_data = []
    
    for input_file in input_files:
        print(f"Loading {input_file}...")
        data = load_data(input_file)
        all_data.extend(data)
        print(f"  Loaded {len(data)} samples")
    
    print(f"\nTotal samples before merge: {len(all_data)}")
    
    # 可选：打乱顺序
    if shuffle:
        random.seed(42)  # 固定随机种子，便于复现
        random.shuffle(all_data)
        print("Data shuffled")
    
    # 保存
    save_data(all_data, output_file, format=output_format)
    
    print(f"\nMerged dataset saved to: {output_file}")
    print(f"Total samples: {len(all_data)}")

if __name__ == "__main__":
    # # ========== 1. 合并 Validation 数据集 ==========
    # validation_files = [
    #     "/root/llama_fine/data_preprocess/validation_task/UTD-MHAD/validation_3MB.json",
    #     "/root/llama_fine/data_preprocess/validation_task/UTD-MHAD/validation_3MB(1).json",
    #     "/root/llama_fine/data_preprocess/validation_task/UTD-MHAD/validation_5MB.json",
    #     "/root/llama_fine/data_preprocess/validation_task/UTD-MHAD/validation_8MB.json",
    #     "/root/llama_fine/data_preprocess/validation_task/UTD-MHAD/validation_10MB.json"
    # ]
    
    # print("=" * 60)
    # print("Merging Validation Datasets")
    # print("=" * 60)
    # merge_datasets(
    #     validation_files,
    #     "/root/llama_fine/data_preprocess/merged/UTD-MHAD/validation_all.json",
    #     shuffle=True
    # )
    
    # # ========== 2. 合并 Repair 数据集 ==========
    # repair_files = [
    #     "/root/llama_fine/data_preprocess/fix_task/UTD-MHAD/repair_3MB.json",
    #     "/root/llama_fine/data_preprocess/fix_task/UTD-MHAD/repair_5MB.json",
    #     "/root/llama_fine/data_preprocess/fix_task/UTD-MHAD/repair_8MB.json",
    #     "/root/llama_fine/data_preprocess/fix_task/UTD-MHAD/repair_10MB.json",
    #     "/root/llama_fine/data_preprocess/fix_task/UTD-MHAD/repair_3MB(1).json"
    # ]
    
    # print("\n" + "=" * 60)
    # print("Merging Repair Datasets")
    # print("=" * 60)
    # merge_datasets(
    #     repair_files,
    #     "/root/llama_fine/data_preprocess/merged/UTD-MHAD/repair_all.json",
    #     shuffle=True
    # )
    
    # # ========== 3. 合并 Improvement 数据集 ==========
    # improvement_files = [
    #     "/root/llama_fine/data_preprocess/imporve_task/UTD-MHAD/improvement_3MB.json",
    #     "/root/llama_fine/data_preprocess/imporve_task/UTD-MHAD/improvement_3MB(1).json",
    #     "/root/llama_fine/data_preprocess/imporve_task/UTD-MHAD/improvement_5MB.json",
    #     "/root/llama_fine/data_preprocess/imporve_task/UTD-MHAD/improvement_8MB.json",
    #     "/root/llama_fine/data_preprocess/imporve_task/UTD-MHAD/improvement_10MB.json"
    # ]
    
    # print("\n" + "=" * 60)
    # print("Merging Improvement Datasets")
    # print("=" * 60)
    # merge_datasets(
    #     improvement_files,
    #     "/root/llama_fine/data_preprocess/merged/UTD-MHAD/improvement_all.json",
    #     shuffle=True
    # )
    
    # ========== 4. 合并 Generation 数据集 ==========
    # 如果 generation 也有多个文件的话
    generation_files = [
        "/root/llama_fine/data_preprocess/fin_gen_task/UTD-MHAD/nas_generation_3MB.jsonl",
        "/root/llama_fine/data_preprocess/fin_gen_task/UTD-MHAD/nas_generation_5MB.jsonl",
        "/root/llama_fine/data_preprocess/fin_gen_task/UTD-MHAD/nas_generation_8MB.jsonl",
        "/root/llama_fine/data_preprocess/fin_gen_task/UTD-MHAD/nas_generation_10MB.jsonl",
        "/root/llama_fine/data_preprocess/fin_gen_task/UTD-MHAD/nas_generation_3MB(1).jsonl"
    ]
    
    print("\n" + "=" * 60)
    print("Merging Generation Datasets")
    print("=" * 60)
    merge_datasets(
        generation_files,
        "/root/llama_fine/data_preprocess/merged/UTD-MHAD/generation_all.json",
        shuffle=True
    )
    
    # # ========== 5. 合并 Stage1: Validation + Repair ==========
    # print("\n" + "=" * 60)
    # print("Merging Stage1: Validation + Repair")
    # print("=" * 60)
    # merge_datasets(
    #     [
    #         "/root/llama_fine/data_preprocess/merged/UTD-MHAD/validation_all.json",
    #         "/root/llama_fine/data_preprocess/merged/UTD-MHAD/repair_all.json"
    #     ],
    #     "/root/llama_fine/data_preprocess/merged/UTD-MHAD/stage1_val_repair.json",
    #     shuffle=True
    # )
    
    # print("\n" + "=" * 60)
    # print("All Done!")
    # print("=" * 60)

    # # ========== 6. 合并 Augmented Generation  ==========

    # generation_files = [
    #     "/root/llama_fine/data_preprocess/generation_aug_task/generation_12MB.jsonl",
    #     "/root/llama_fine/data_preprocess/generation_aug_task/generation_15MB.jsonl",
    #     "/root/llama_fine/data_preprocess/generation_aug_task/generation_20MB.jsonl"
    # ]
    
    # print("\n" + "=" * 60)
    # print("Merging Generation Datasets")
    # print("=" * 60)
    # merge_datasets(
    #     generation_files,
    #     "/root/llama_fine/data_preprocess/merged/aug_generation_all.json",
    #     shuffle=True
    # )

    # ========== 7. 合并 Augmented Generation 2  ==========

    # generation_files = [
    #     "/root/llama_fine/data_preprocess/generation_aug_2/generation.jsonl"
    # ]
    
    # print("\n" + "=" * 60)
    # print("Merging Generation Datasets")
    # print("=" * 60)
    # merge_datasets(
    #     generation_files,
    #     "/root/llama_fine/data_preprocess/merged/2_aug_generation_all.json",
    #     shuffle=True
    # )