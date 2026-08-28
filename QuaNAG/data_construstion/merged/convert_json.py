import json
import sys

def convert_to_llamafactory_format(input_file, output_file):
    """
    将数据格式转换为 LlamaFactory Alpaca 格式
    
    input/output中的 dict 转为 JSON 字符串
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted = []
    
    for item in data:
        # 转换 input 为字符串
        input_content = item.get('input', '')
        if isinstance(input_content, dict):
            input_str = json.dumps(input_content, ensure_ascii=False)
        elif isinstance(input_content, str):
            input_str = input_content
        else:
            input_str = str(input_content)
        
        # 转换 output 为字符串
        output_content = item.get('output', '')
        if isinstance(output_content, dict):
            output_str = json.dumps(output_content, ensure_ascii=False)
        elif isinstance(output_content, str):
            output_str = output_content
        else:
            output_str = str(output_content)
        
        converted_item = {
            'instruction': item.get('instruction', ''),
            'input': input_str,
            'output': output_str
        }
        
        # 保留其他字段（如 dialogue_id, _metadata 等）
        for key in item:
            if key not in ['instruction', 'input', 'output']:
                converted_item[key] = item[key]
        
        converted.append(converted_item)
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(converted)} samples")
    print(f"   Input:  {input_file}")
    print(f"   Output: {output_file}")

if __name__ == "__main__":
    # 转换所有数据集
    datasets = [
        # ("/root/llama_fine/data_preprocess/merged/UTD-MHAD/stage1_val_repair.json",
        #  "/root/llama_fine/data_preprocess/merged/UTD-MHAD/stage1_val_repair_converted.json")
        
        # ("/root/llama_fine/data_preprocess/merged/USCHAD/improvement_aug.json",
        #  "/root/llama_fine/data_preprocess/merged/USCHAD/improvement_aug_converted.json"),
        
        # ("/root/llama_fine/data_preprocess/merged/UTD-MHAD/improvement_aug.json",
        #  "/root/llama_fine/data_preprocess/merged/UTD-MHAD/improvement_aug_converted.json"),
        
        # ("/root/llama_fine/data_preprocess/merged/Wharf1/generation_aug.json",
        #  "/root/llama_fine/data_preprocess/merged/Wharf1/generation_aug_converted.json"),
        ("/root/llama_fine/data_preprocess/merged/UTD-MHAD/generation_all.json",
         "/root/llama_fine/data_preprocess/merged/UTD-MHAD/generation_all_converted.json")
        # ("/root/llama_fine/data_preprocess/merged/2_aug_generation_all.json",
        #  "/root/llama_fine/data_preprocess/merged/2_aug_generation_all_converted.json")
        
    ]
    
    print("=" * 60)
    print("Converting datasets to LlamaFactory format")
    print("=" * 60)
    print()
    
    for input_file, output_file in datasets:
        try:
            convert_to_llamafactory_format(input_file, output_file)
            print()
        except FileNotFoundError:
            print(f"⚠️  File not found: {input_file}")
            print()
        except Exception as e:
            print(f"❌ Error converting {input_file}: {e}")
            print()
    
    print("=" * 60)
    print("Conversion completed!")
    print("=" * 60)