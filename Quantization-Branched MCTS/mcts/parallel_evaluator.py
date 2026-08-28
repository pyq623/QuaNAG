"""
统一的并行模型评估器 (Parallel Model Evaluator)
支持 HAR (1D) 和 Image (2D) 任务

工作进程使用此评估器进行模型训练和量化评估
"""
from typing import List, Dict, Any, Optional, Tuple
from queue import Empty
import time
import copy
import torch
import torch.multiprocessing as mp

# 统一导入
from data import get_dataset_info, get_data_loader
from models import CandidateModel, QuantizableModel
from models import get_quantization_option, apply_configurable_static_quantization
from models import fuse_QATmodel_modules, prepare_qaft_model, apply_qaft_quantization
from utils import calculate_memory_usage


class ParallelModelEvaluator:
    """
    统一的并行模型评估器
    
    支持:
    - HAR任务 (1D时序数据)
    - Image任务 (2D图像数据)
    
    每个工作进程有一个独立的评估器实例
    """
    
    def __init__(self, gpu_id: int, constraints: Dict[str, float],
                 dataset_name: str, train_epochs: int = 100):
        """
        初始化评估器
        
        Args:
            gpu_id: GPU设备ID
            constraints: 约束条件
            dataset_name: 数据集名称（会自动推断任务类型）
            train_epochs: 训练轮数
        """
        self.gpu_id = gpu_id
        self.constraints = constraints
        self.dataset_name = dataset_name
        self.train_epochs = train_epochs
        
        # 设置设备
        self.device = f"cuda:{gpu_id}"
        
        # ✅ 获取数据集信息（自动推断任务类型）
        self.dataset_info = get_dataset_info(dataset_name)
        self.task_type = self.dataset_info.get('task_type', 'har')
        
        # ✅ 使用统一的数据加载器
        data_loader = get_data_loader(self.task_type)
        
        # ✅ 根据任务类型设置数据路径
        if self.task_type == 'har':
            root_dir = "/root/unified_nas/data/UniMTS_data"
            batch_size = 64
        else:  # image
            root_dir = "/root/unified_nas/data"
            batch_size = 128
        
        # ✅ 加载数据集
        dataloaders = data_loader.get_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            datasets_list=[dataset_name],
            num_workers=0
        )
        self.dataloader = dataloaders[dataset_name]
        
        print(f"[GPU {gpu_id}] 评估器初始化完成")
        print(f"  - 任务类型: {self.task_type.upper()}")
        print(f"  - 数据集: {dataset_name}")
    
    def evaluate(self, config: Dict[str, Any], candidate_id: str) -> \
            Tuple[float, Dict[str, Any]]:
        """
        评估单个配置
        
        Args:
            config: 模型配置
            candidate_id: 候选ID
        
        Returns:
            (奖励值, 指标字典)
        """
        try:
            print(f"[GPU {self.gpu_id}] 开始评估 {candidate_id}")
            
            # 1. 创建候选模型（传递task_type）
            candidate = CandidateModel(
                config=config,
                task_type=self.task_type  # ✅ 传递任务类型
            )
            candidate.candidate_id = candidate_id
            
            # 2. 获取量化模式
            quant_mode = config.get('quant_mode', 'none')
            self._current_quant_mode = quant_mode
            print(f"[GPU {self.gpu_id}] 量化模式: {quant_mode}")
            
            # 3. 构建模型
            model = candidate.build_model()

            # ✅ 根据任务类型构造输入尺寸
            if self.task_type == 'har':
                # HAR: (batch, channels, time_steps)
                input_size = (
                    64,
                    self.dataset_info['channels'],
                    self.dataset_info['time_steps']
                )
            else:  # image
                # Image: (batch, channels, height, width)
                input_size = (
                    1,
                    self.dataset_info['channels'],
                    self.dataset_info['height'],
                    self.dataset_info['width']
                )
                
            memory_usage = calculate_memory_usage(
                model,
                input_size=input_size,
                device='cpu'
            )
            print(f"[GPU {self.gpu_id}] 原始内存: "
                  f"{memory_usage['total_memory_MB']:.2f}MB")
            
            # ========== 公平的训练策略 ==========
            
            if quant_mode == 'qaft':
                # QAFT: 两阶段训练
                print(f"[GPU {self.gpu_id}] ⚖️ QAFT 模式 - 两阶段训练 "
                      f"(总计{self.train_epochs}轮)")
                
                # 计算轮数分配 (90% + 10%)
                pretrain_epochs = int(self.train_epochs * 0.9)
                finetune_epochs = self.train_epochs - pretrain_epochs
                
                print(f"[GPU {self.gpu_id}] 📊 训练分配:")
                print(f"  - 阶段1(预训练): {pretrain_epochs}轮")
                print(f"  - 阶段2(微调): {finetune_epochs}轮")
                print(f"  - 总计: {pretrain_epochs + finetune_epochs}轮")
                
                # 阶段1: 正常训练预热
                print(f"[GPU {self.gpu_id}] 🔥 阶段1: 预训练开始...")
                from training import SingleTaskTrainer
                
                pretrain_trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                pretrain_accuracy, pretrain_metrics, _, pretrain_state = \
                    pretrain_trainer.train(epochs=pretrain_epochs)
                
                print(f"[GPU {self.gpu_id}] ✅ 阶段1完成: "
                      f"准确率={pretrain_accuracy:.2f}%")
                
                # 阶段2: QAFT 微调
                print(f"[GPU {self.gpu_id}] 🎯 阶段2: QAFT微调开始...")
                
                # 加载最佳预训练权重
                if pretrain_state and 'model' in pretrain_state:
                    model.load_state_dict(pretrain_state['model'])
                    print(f"[GPU {self.gpu_id}] 📥 已加载预训练权重")
                
                # 准备QAFT (插入FakeQuantize)
                model = self._prepare_qat_model(model)
                
                # 微调训练
                qaft_trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                best_accuracy, best_val_metrics, history, best_model_state = \
                    qaft_trainer.train(
                        epochs=finetune_epochs,
                        save_path=f"checkpoints/candidate_{candidate_id}.pth"
                    )
                
                print(f"[GPU {self.gpu_id}] ✅ 阶段2完成: "
                      f"准确率={best_accuracy:.2f}%")
                print(f"[GPU {self.gpu_id}] 📈 准确率变化: "
                      f"{pretrain_accuracy:.2f}% → {best_accuracy:.2f}%")
            
            elif quant_mode == 'qat':
                # QAT: 插入量化后完整训练
                print(f"[GPU {self.gpu_id}] ⚡ QAT模式 - 量化感知训练 "
                      f"({self.train_epochs}轮)")
                model = self._prepare_qat_model(model)
                
                from training import SingleTaskTrainer
                trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                best_accuracy, best_val_metrics, history, best_model_state = \
                    trainer.train(epochs=self.train_epochs)
                
                print(f"[GPU {self.gpu_id}] ✅ QAT 训练完成: "
                      f"准确率={best_accuracy:.2f}%")
            
            else:
                # None/Static: 正常训练
                print(f"[GPU {self.gpu_id}] 🔄 {quant_mode.upper()}模式 - "
                      f"正常训练 ({self.train_epochs}轮)")
                from training import SingleTaskTrainer
                trainer = SingleTaskTrainer(
                    model=model,
                    dataloaders=self.dataloader,
                    device=self.device
                )
                
                best_accuracy, best_val_metrics, history, best_model_state = \
                    trainer.train(
                        epochs=self.train_epochs,
                        save_path=f"checkpoints/candidate_{candidate_id}.pth"
                    )
                
                print(f"[GPU {self.gpu_id}] ✅ 训练完成: "
                      f"准确率={best_accuracy:.2f}%")
            
            # ========== 测量原始模型性能 ==========
            
            # ✅ 根据任务类型构造输入尺寸
            if self.task_type == 'har':
                # HAR: (batch, channels, time_steps)
                input_size = (
                    64,
                    self.dataset_info['channels'],
                    self.dataset_info['time_steps']
                )
            else:  # image
                # Image: (batch, channels, height, width)
                input_size = (
                    1,
                    self.dataset_info['channels'],
                    self.dataset_info['height'],
                    self.dataset_info['width']
                )
            
            # memory_usage = 
            calculate_memory_usage(
                model,
                input_size=input_size,
                device='cpu'
            )
            # print(f"[GPU {self.gpu_id}] 原始内存: "
            #       f"{memory_usage['total_memory_MB']:.2f}MB")
            
            # 初始奖励和准确率
            reward = best_accuracy
            accuracy = best_accuracy
            
            metrics = {
                "original_accuracy": accuracy,
                "original_accuracy_percent": best_accuracy,
                "original_memory": memory_usage['total_memory_MB'],
                "reward": reward,
                "train_loss": best_val_metrics.get('loss', 0.0) \
                             if best_val_metrics else 0.0,
                "gpu_id": self.gpu_id,
                "quantization_mode": quant_mode,
                "task_type": self.task_type  # ✅ 记录任务类型
            }
            
            # ========== 量化处理 ==========
            
            if quant_mode != 'none':
                quant_accuracy, quant_metrics = self._apply_quantization(
                    model, best_model_state, quant_mode, candidate_id,
                    input_size  # ✅ 传递输入尺寸
                )
                
                accuracy_drop = best_accuracy - quant_accuracy
                
                # 更新指标
                metrics.update({
                    'quantized_accuracy': quant_metrics.get('accuracy', 0),
                    'quantized_memory': quant_metrics.get(
                        'peak_memory', memory_usage['total_memory_MB']
                    ),
                    'quantization_method': quant_metrics.get('method', 'unknown'),
                    'accuracy_drop': accuracy_drop
                })
                
                # 量化后使用量化准确率作为奖励
                reward = quant_accuracy
                metrics['accuracy'] = quant_accuracy
                
                print(f"[GPU {self.gpu_id}] 使用量化奖励: {reward:.2f}")
                print(f"  - 原始准确率: {best_accuracy:.2f}%")
                print(f"  - 量化准确率: {quant_accuracy:.2f}%")
                print(f"  - 准确率下降: {accuracy_drop:.2f}%")
            else:
                reward = best_accuracy
                metrics['accuracy'] = best_accuracy
                print(f"[GPU {self.gpu_id}] 未进行量化，使用原始奖励: {reward:.2f}")
            
            metrics['reward'] = reward
            print(f"[GPU {self.gpu_id}] 完成评估 {candidate_id}: "
                  f"奖励={reward:.2f}")
            
            save_path = f"checkpoints/candidate_{candidate_id}_full.pth"
            torch.save({
                'config': config,
                'model': best_model_state['model'] if best_model_state else None,
                'head': best_model_state['head'] if best_model_state else None,
                'task_type': self.task_type,
                'dataset_name': self.dataset_name,
                'quant_mode': quant_mode,
                'metrics': metrics
            }, save_path)
            print(f"[GPU {self.gpu_id}] 💾 完整模型已保存: {save_path}")
            
            return reward, metrics
        
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 评估失败 {candidate_id}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {
                "error": str(e),
                "accuracy": 0.0,
                "reward": 0.0,
                "gpu_id": self.gpu_id,
                "task_type": self.task_type
            }
    
    def _prepare_qat_model(self, model):
        """准备QAT量化感知训练模型"""
        try:
            # 获取量化模式
            quant_mode = getattr(self, '_current_quant_mode', 'qat')
            print(f"\n======\n prepare quant_mode:{quant_mode}\n======\n")
            if quant_mode == 'qaft':
                print(f"[GPU {self.gpu_id}] 准备QAFT量化 (微调模式)")
                model = prepare_qaft_model(model, freeze_backbone=True)
            else:
                print(f"[GPU {self.gpu_id}] 准备QAT量化 (完整训练)")
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
                fuse_QATmodel_modules(model)
                model.train()
                torch.quantization.prepare_qat(model, inplace=True)
            
            print(f"[GPU {self.gpu_id}] 量化准备完成")
            return model
        
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 量化准备失败: {e}")
            import traceback
            traceback.print_exc()
            return model
    
    def _apply_quantization(self, model, best_state: dict, quant_mode: str,
                           candidate_id: str, input_size: tuple) -> \
            Tuple[float, Dict[str, Any]]:
        """
        应用量化并评估
        
        Args:
            model: 训练好的模型
            best_state: 最佳状态字典
            quant_mode: 量化模式
            candidate_id: 候选ID
            input_size: 输入尺寸（用于内存计算）
        
        Returns:
            (量化准确率, 量化指标字典)
        """
        try:
            print(f"[GPU {self.gpu_id}] 应用{quant_mode}量化")
            
            # 根据量化模式选择配置
            if quant_mode == 'static':
                quantization_options = [
                    ('int8_default', '默认INT8量化'),
                    ('int8_per_channel', '逐通道INT8量化'),
                    ('int8_reduce_range', '减少范围INT8量化'),
                    ('int8_asymmetric', 'INT8非对称量化'),
                    ('int8_histogram', 'INT8直方图校准'),
                    ('int8_moving_avg', 'INT8移动平均校准')
                ]
            elif quant_mode == 'qat':
                quantization_options = [('qat_default', 'QAT量化')]
            elif quant_mode == 'qaft':
                print(f"[GPU {self.gpu_id}] QAFT模式 - 直接转换")
                quantization_options = [('qaft_default', 'QAFT量化')]
            else:
                return 0.0, {}
            
            best_accuracy = 0.0
            best_quantized_model = None
            best_memory = 0.0
            best_option_name = ""
            
            # 尝试不同量化配置
            for option_name, option_desc in quantization_options:
                try:
                    print(f"[GPU {self.gpu_id}] 尝试 {option_desc}")
                    
                    quantized_model = self._apply_quantization_helper(
                        model, quant_mode, option_name
                    )
                    
                    if quantized_model:
                        # 创建任务头并加载权重
                        import torch.nn as nn
                        task_head = nn.Linear(
                            model.output_dim,
                            len(self.dataloader['test'].dataset.classes)
                        ).to('cpu')
                        
                        if best_state and 'head' in best_state:
                            task_head.load_state_dict(best_state['head'])
                        
                        # 评估量化模型
                        from models import evaluate_quantized_model
                        quant_accuracy = evaluate_quantized_model(
                            quantized_model, self.dataloader, task_head,
                            f"量化模型({option_name})"
                        )
                        
                        # ✅ 测量内存（使用传入的input_size）
                        quant_memory = calculate_memory_usage(
                            quantized_model,
                            input_size=input_size,
                            device='cpu'
                        )['total_memory_MB']
                        
                        print(f"[GPU {self.gpu_id}] 📊 {option_desc}: "
                              f"{quant_accuracy:.1f}% / {quant_memory:.2f}MB")
                        
                        # 记录最佳结果
                        if quant_accuracy > best_accuracy:
                            best_accuracy = quant_accuracy
                            best_quantized_model = quantized_model
                            best_memory = quant_memory
                            best_option_name = option_name

                except Exception as e:
                    print(f"[GPU {self.gpu_id}] {option_desc} 失败: {e}")
                    continue
            
            # 保存最佳量化模型
            if best_quantized_model:
                print(f"[GPU {self.gpu_id}] 最佳量化: {best_option_name}, "
                      f"准确率={best_accuracy:.1f}%")
                
                # ✅ 新增：保存量化模型
                quant_save_path = f"checkpoints/quantized_{candidate_id}.pth"
                torch.save({
                    'quantized_model': best_quantized_model.state_dict(),
                    'quant_method': best_option_name,
                    'quant_accuracy': best_accuracy,
                    'quant_memory': best_memory
                }, quant_save_path)
                print(f"[GPU {self.gpu_id}] 💾 量化模型已保存: {quant_save_path}")

                return best_accuracy, {
                    'accuracy': best_accuracy,
                    'method': best_option_name,
                    'peak_memory': best_memory
                }
            
            return 0.0, {}
        
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 量化失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {}
    
    def _apply_quantization_helper(self, model, quant_mode: str,
                                   quantization_option: str = 'int8_per_channel'):
        """量化辅助方法"""
        model_copy = copy.deepcopy(model)
        model_copy.to('cpu').eval()
        # model_copy.eval()
        # model_copy.to('cpu')
        
        if quant_mode == 'static':
            # 静态量化
            quant_config = get_quantization_option(quantization_option)
            print(f"[GPU {self.gpu_id}] 量化配置: "
                  f"{quant_config['description']}")
            
            quantized_model = apply_configurable_static_quantization(
                model_copy,
                self.dataloader,
                precision=quant_config['precision'],
                backend=quant_config['backend']
            )
        
        elif quant_mode in ['qat', 'qaft']:
            # QAT和QAFT都使用convert转换
            print(f"[GPU {self.gpu_id}] 转换{quant_mode.upper()}模型")
            # model_copy.eval()
            quantized_model = torch.quantization.convert(model_copy, inplace=False)
        
        else:
            print(f"[GPU {self.gpu_id}] 未知量化模式: {quant_mode}")
            return model_copy
        
        return quantized_model


def worker_process(worker_id: int, gpu_id: int,
                   task_queue: mp.Queue, result_queue: mp.Queue,
                   constraints: Dict[str, float], dataset_name: str,
                   train_epochs: int):
    """
    工作进程函数
    
    Args:
        worker_id: 工作进程ID (0-11)
        gpu_id: 实际GPU ID (0-3，支持复用)
        task_queue: 任务队列（接收配置）
        result_queue: 结果队列（返回评估结果）
        constraints: 约束条件
        dataset_name: 数据集名称（会自动推断任务类型）
        train_epochs: 训练轮数
    """
    print(f"[Worker-{worker_id}] 启动，使用 GPU{gpu_id}")
    
    # 创建评估器
    evaluator = ParallelModelEvaluator(
        gpu_id=gpu_id,
        constraints=constraints,
        dataset_name=dataset_name,
        train_epochs=train_epochs
    )
    
    print(f"[Worker-{worker_id}] 评估器初始化完成，任务类型: "
          f"{evaluator.task_type.upper()}")
    
    while True:
        try:
            # 从队列获取任务（超时60秒）
            task = task_queue.get(timeout=60)
            
            # 检查是否是终止信号
            if task is None:
                print(f"[Worker-{worker_id}] 收到终止信号")
                break
            
            # 解包任务
            candidate_id, config = task
            
            # 评估
            reward, metrics = evaluator.evaluate(config, candidate_id)
            
            # 返回结果
            result_queue.put((candidate_id, reward, metrics))
        
        except Empty:
            # 队列超时，继续等待
            continue
        
        except Exception as e:
            print(f"[Worker-{worker_id}] 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[Worker-{worker_id}] 退出")