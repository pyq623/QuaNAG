"""
简单的LLaMA配置生成服务器
运行在 llama_factory 环境
"""
import zmq
import json
import re
import time
from llamafactory.chat import ChatModel

class SimpleLLaMAServer:
    def __init__(self, port=5555):
        self.port = port
        
        # 初始化模型
        print("🔧 加载 LLaMA3-8B 模型...")
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-11-24-13-28-04
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-09-08-50-50
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-07-08-58-38
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-11-09-24-02
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-12-11-33-54    # Mhealth generation
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-16-01-32-31
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-21-12-16-02
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-23-11-30-42
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-24-08-23-29
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2025-12-25-05-57-23
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-03-15-22-33 Mhealth DPO
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-07-09-46-43 MMAct1 generation
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-07-12-41-41 MMAct1 dpo
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-13-00-30-03 Wharf generation
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-13-02-17-52 Wharf dpo
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-14-08-34-07 USCHAD generation
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-13-09-20-52 USCHAD dpo
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-14-09-33-03 UTD-MHAD dpo
        # /root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-18-10-00-00 UTD-MHAD dpo Failed retry
        args = {
            "model_name_or_path": "/root/llama_fine/llama3.1-8b",  # ✅ 基础模型
            "adapter_name_or_path": "/root/llama_fine/LLaMA-Factory/saves/Llama-3.1-8B/lora/train_2026-01-03-15-22-33",  # ✅ LoRA adapter（在LLaMA-Factory目录下）
            "template": "llama3",
            "finetuning_type": "lora"
        }
        self.chat_model = ChatModel(args)
        print("✅ 模型加载完成!\n")
        self.check_device()  # 添加这行
        
        # 初始化ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"🚀 服务器启动: tcp://localhost:{port}")
        print("⏳ 等待请求...\n")
    
    # def extract_json(self, text):
    #     """提取JSON配置"""
    #     try:
    #         # 尝试直接解析
    #         json_match = re.search(r'\{[\s\S]*\}', text)
    #         if json_match:
    #             return json.loads(json_match.group())
            
    #         # 尝试从代码块提取
    #         code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    #         if code_match:
    #             return json.loads(code_match.group(1))
            
    #         return None
    #     except:
    #         return None
    def clean_model_output(self, text):
        """清理模型输出，提取JSON"""
        # 方法1：找到第一个完整的JSON对象
        try:
            # 找到第一个{和最后一个}
            start = text.find('{')
            if start == -1:
                return None
            
            # 从start开始，找到匹配的}
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到了完整的 JSON
                        json_str = text[start:i+1]
                        return json.loads(json_str)
            
            return None
        except:
            return None
    def extract_json(self, text):
        """提取第一个 ```json 代码块中的内容"""
        try:
            # 查找第一个 ```json
            start_marker = "```json"
            start_idx = text.find(start_marker)
            
            if start_idx == -1:
                # 如果没有 ```json，尝试查找普通的 ```
                start_marker = "```"
                start_idx = text.find(start_marker)
                if start_idx == -1:
                    return None
            
            # 跳过起始标记
            content_start = start_idx + len(start_marker)
            
            # 从起始标记之后查找结束的 ```
            end_idx = text.find("```", content_start)
            
            if end_idx == -1:
                return None
            
            # 提取JSON字符串
            json_str = text[content_start:end_idx].strip()
            
            # 解析JSON
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析错误: {e}")
            return None
        except Exception as e:
            print(f"⚠️ 提取错误: {e}")
            return None
    
    def run(self):
        """主循环"""
        count = 0
        try:
            while True:
                # 接收请求
                request = self.socket.recv_json()
                count += 1
                
                print(f"\n{'='*60}")
                print(f"📨 请求 #{count}")
                
                # 检查关闭信号
                if request.get("command") == "shutdown":
                    self.socket.send_json({"success": True})
                    print("🛑 收到关闭信号")
                    break
                
                prompt = request.get("prompt", "")
                if not prompt:
                    self.socket.send_json({
                        "success": False,
                        "error": "No prompt"
                    })
                    continue
                
                print(f"📝 Prompt长度: {len(prompt)} 字符")
                
                # 生成配置
                messages = [{"role": "user", "content": prompt}]
                start_time = time.time()
                response = self.chat_model.chat(messages)
                response_text = response[0].response_text
                elapsed = time.time() - start_time
                print(f"🤖 响应: {response_text}\n")  # 只打印前100个字符
                print(f"🤖 响应长度: {len(response_text)} 字符")
                print(f"⏱️ 生成耗时: {elapsed:.1f} 秒")
                
                # 解析JSON
                config = self.clean_model_output(response_text)
                # config = self.extract_json(response_text)
                
                if config:
                    print(f"✅ 成功解析配置")
                    print(f"   - 输入通道: {config.get('input_channels')}")
                    print(f"   - 类别数: {config.get('num_classes')}")
                    print(f"   - 量化模式: {config.get('quant_mode')}")
                    
                    self.socket.send_json({
                        "success": True,
                        "config": config
                    })
                else:
                    print(f"❌ JSON解析失败")
                    self.socket.send_json({
                        "success": False,
                        "error": "Failed to parse JSON",
                        "raw_response": response_text[:500]
                    })
                
        except KeyboardInterrupt:
            print("\n⚠️ 收到中断信号")
        finally:
            print(f"\n📊 共处理 {count} 个请求")
            self.socket.close()
            self.context.term()
            print("✅ 服务器已关闭")
    def check_device(self):
        """检查模型实际运行在哪里"""
        import torch
        print("\n" + "="*60)
        print("🔍 设备检查")
        print("="*60)
        
        try:
            # 检查模型
            model = self.chat_model.engine.model
            device = next(model.parameters()).device
            print(f"📍 模型在: {device}")
            
            if device.type == 'cpu':
                print("⚠️⚠️⚠️ 警告：模型在CPU上运行！ ⚠️⚠️⚠️")
            else:
                print(f"✅ 模型在 GPU {device.index} 上")
                
            # 检查CUDA可用性
            print(f"\n🎮 CUDA状态:")
            print(f"   - CUDA可用: {torch.cuda.is_available()}")
            print(f"   - CUDA版本: {torch.version.cuda}")
            print(f"   - GPU数量: {torch.cuda.device_count()}")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"     已用内存: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")
        except Exception as e:
            print(f"❌ 检查失败: {e}")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    server = SimpleLLaMAServer(port=5555)
    server.run()