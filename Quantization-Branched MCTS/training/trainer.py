import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import copy

# 设置随机数种子
SEED = 42  # 你可以选择任何整数作为种子
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class SingleTaskTrainer:
    """
    针对单个数据集的训练器
    """
    def __init__(self, model, dataloaders, device='cuda', logger=None):
        """
        初始化训练器

        参数:
            model: 要训练的模型
            dataloaders: 数据加载器字典，包含 'train' 和 'test' 两个键
            device: 训练设备 ('cuda' 或 'cpu')
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        self.logger = logger

        # 如果没有提供logger，创建一个简单的logger来模拟print行为

        # 确保模型有 output_dim 属性
        if not hasattr(model, 'output_dim'):
            raise AttributeError("Model must have 'output_dim' attribute")

        # 获取类别数
        self.num_classes = len(dataloaders['train'].dataset.classes)
        print(f"Number of classes: {self.num_classes}")

        # 创建任务头
        self.task_head = nn.Linear(model.output_dim, self.num_classes).to(device)

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            list(model.parameters()) + list(self.task_head.parameters()),
            lr=1e-3
        )

    def _output(self, message):
        """统一的输出方法：如果有logger则使用logger，否则使用print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_epoch(self):
        """
        单个训练周期
        """
        self.model.train()
        self.task_head.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.dataloaders['train'], desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            features = self.model(inputs)
            outputs = self.task_head(features)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        metrics = {
            'loss': running_loss / len(self.dataloaders['train']),
            'accuracy': 100. * correct / total
        }
        return metrics

    def evaluate(self):
        """
        模型评估
        """
        self.model.eval()
        self.task_head.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['test'], desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                features = self.model(inputs)
                outputs = self.task_head(features)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        metrics = {
            'loss': running_loss / len(self.dataloaders['test']),
            'accuracy': 100. * correct / total
        }
        return metrics

    def train(self, epochs=10, save_path='best_model.pth'):
        """
        训练模型并保存最佳权重

        参数:
            epochs: 训练周期数
            save_path: 最佳模型权重保存路径
        返回:
            best_accuracy: 最佳验证准确率
            best_val_metrics: 最佳验证指标
            history: 训练历史记录
            best_model_state: 最佳模型状态字典
        """
        best_accuracy = 0.0
        best_val_metrics = None  # 保存最佳验证指标
        history = []
        best_model_state = None  # 保存最佳模型状态

        for epoch in range(epochs):
            # print(f"\nEpoch {epoch + 1}/{epochs}")
            self._output(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练阶段
            train_metrics = self.train_epoch()

            # 验证阶段
            val_metrics = self.evaluate()

            # 保存历史
            history.append({
                'train': train_metrics,
                'val': val_metrics
            })

            # print(f"\nValidation Accuracy: {val_metrics['accuracy']:.2f}%")
            self._output(f"\nValidation Accuracy: {val_metrics['accuracy']:.2f}%")

            # 更新最佳模型
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                best_val_metrics = val_metrics
                best_model_state = {
                    # 'model': self.model.state_dict(),
                    # 'head': self.task_head.state_dict()
                    'model': copy.deepcopy(self.model.state_dict()),  # ✅ 深拷贝
                    'head': copy.deepcopy(self.task_head.state_dict())  # ✅ 深拷贝
                }

                # 保存最佳模型权重到文件
                torch.save(best_model_state, save_path)
                # print(f"✅ Best model saved with accuracy: {best_accuracy:.2f}%")
                self._output(f"✅ Best model saved with accuracy: {best_accuracy:.2f}%")

        return best_accuracy, best_val_metrics, history, best_model_state

class MultiTaskTrainer:
    def __init__(self, model, dataloaders, device='cuda'):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        
        self.device = device

        # 确保模型有output_dim属性
        if not hasattr(model, 'output_dim'):
            raise AttributeError("Model must have 'output_dim' attribute")

        
        # 动态创建多任务头（根据数据集类别数）
        self.task_heads = nn.ModuleDict()
        for task_name, loaders in dataloaders.items():
            if loaders['train'] is None:
                continue
                
            # 从数据集获取类别数
            num_classes = len(loaders['train'].dataset.classes)
            print(f"Creating task head for {task_name}: input_dim={model.output_dim}, output_dim={num_classes}")
            self.task_heads[task_name] = nn.Linear(model.output_dim, num_classes).to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            list(model.parameters()) + list(self.task_heads.parameters()),
            lr=1e-3
        )
        
    def train_epoch(self):
        self.model.train()
        for head in self.task_heads.values():
            head.train()
            
        task_metrics = defaultdict(dict)
        
        # 交替训练不同数据集
        for task_name, loaders in self.dataloaders.items():
            if loaders['train'] is None:
                continue
                
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(loaders['train'], desc=f"Training {task_name}"):
                inputs = inputs.to(self.device)  # [B, C, T]
                labels = labels.to(self.device)

                # Add debug print to check shapes
                # print(f"Input shape: {inputs.shape}")  # Debug
                
                self.optimizer.zero_grad()
                features = self.model(inputs)

                # Add debug print for features shape
                # print(f"Features shape: {features.shape}")  # Debug

                outputs = self.task_heads[task_name](features)

                # Add debug print for outputs shape
                # print(f"Outputs shape: {outputs.shape}")  # Debug
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            task_metrics[task_name] = {
                'loss': running_loss / len(loaders['train']),
                'accuracy': 100. * correct / total
            }
        
        return task_metrics
    
    def evaluate(self):
        self.model.eval()
        for head in self.task_heads.values():
            head.eval()
            
        task_metrics = defaultdict(dict)
        
        with torch.no_grad():
            for task_name, loaders in self.dataloaders.items():
                if loaders['test'] is None:
                    continue
                    
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in tqdm(loaders['test'], desc=f"Evaluating {task_name}"):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    features = self.model(inputs)
                    outputs = self.task_heads[task_name](features)
                    
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                task_metrics[task_name] = {
                    'loss': running_loss / len(loaders['test']),
                    'accuracy': 100. * correct / total
                }
        
        return task_metrics
    
    def train(self, epochs=10, save_path='best_model.pth', patience=5, min_delta=0.01):
        """
        训练模型并保存最佳权重，支持早停机制。

        参数:
            epochs: 最大训练周期数
            save_path: 最佳模型权重保存路径
            patience: 早停的容忍周期数
            min_delta: 验证指标的最小变化量
        返回:
            best_avg_acc: 最佳平均验证准确率
            best_val_metrics: 最佳验证指标
            history: 训练历史记录
            best_model_state: 最佳模型状态字典
        """
        best_avg_acc = 0.0
        best_val_metrics = None  # 用于保存最佳权重对应的验证准确率
        history = []
        best_model_state = None  # 用于保存最佳模型状态

        # Early stopping variables
        no_improvement_epochs = 0  # 记录验证集指标未提升的连续周期数
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练阶段
            train_metrics = self.train_epoch()
            
            # 验证阶段
            val_metrics = self.evaluate()
            
            # 计算平均准确率
            avg_acc = np.mean([m['accuracy'] for m in val_metrics.values()])
            # print("avg_acc:", avg_acc)
            # 保存历史
            history.append({
                'train': train_metrics,
                'val': val_metrics,
                'avg_acc': avg_acc
            })
            
            print(f"\nValidation Accuracy:")
            for task, metrics in val_metrics.items():
                print(f"{task}: {metrics['accuracy']:.2f}%")
            print(f"Average Accuracy: {avg_acc:.2f}%")
            
            # # 更新最佳模型
            # if avg_acc > best_avg_acc:
            #     best_avg_acc = avg_acc
            #     best_val_metrics = val_metrics  # 保存最佳验证准确率对应的各任务指标
            #     best_model_state = {
            #         'model': self.model.state_dict(),
            #         'heads': self.task_heads.state_dict()
            #     }
            
            # # 保存最佳模型权重到文件
            # torch.save(best_model_state, save_path)
            # print(f"✅ Best model saved with accuracy: {best_avg_acc:.2f}%")
            
            # 检查是否为最佳模型
            if avg_acc > best_avg_acc + min_delta:  # 验证集指标有显著提升
                best_avg_acc = avg_acc
                best_val_metrics = val_metrics  # 保存最佳验证准确率对应的各任务指标
                best_model_state = {
                    'model': self.model.state_dict(),
                    'heads': self.task_heads.state_dict()
                }
                
                # 保存最佳模型权重到文件
                torch.save(best_model_state, save_path)
                print(f"✅ Best model saved with accuracy: {best_avg_acc:.2f}%")

                no_improvement_epochs = 0  # 重置早停计数器
            else:
                no_improvement_epochs += 1  # 验证集指标未提升，增加计数器
                print(f"⚠️ No improvement for {no_improvement_epochs} epoch(s).")

            # 检查是否触发早停
            if no_improvement_epochs >= patience:
                print(f"⏹️ Early stopping triggered. No improvement for {patience} consecutive epochs.")
                break
        
        return best_avg_acc, best_val_metrics, history, best_model_state