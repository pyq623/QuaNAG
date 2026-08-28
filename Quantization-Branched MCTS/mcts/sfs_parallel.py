"""
统一的并行散射森林搜索 (Parallel Scattered Forest Search)
支持 HAR (1D) 和 Image (2D) 任务

增强功能:
1. 统一HAR和Image任务处理
2. 完整的对话记录系统
3. 断点恢复功能
4. 改进的重复配置处理
5. 内存约束自适应
6. 紧急降级策略
"""
import uuid
import json
import hashlib
import numpy as np
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from queue import Empty
import time
import pickle
import os
from datetime import datetime
import sys
import random
import torch

# 导入统一的模块
from mcts.mcts_nodes import MCTSNode, MCTSTree
from mcts.config_generator import ConfigGenerator, ScatteringGenerator
from models import CandidateModel
from llm_prompt import LLMConfigGenerator
from utils import calculate_memory_usage
from data import get_dataset_info, get_data_loader
from mcts import worker_process
from mcts import ConfigDegradationManager


class ParallelScatteredForestSearch:
    """
    统一的并行散射森林搜索
    
    支持:
    - HAR任务 (1D时序数据)
    - Image任务 (2D图像数据)
    
    主进程负责:
      - 维护搜索树
      - 生成配置（LLM调用）
      - 协调全局状态
      - 分配任务到工作进程
      - 断点恢复
      - 对话记录管理
    
    工作进程负责:
      - 训练评估模型
    """
    
    def __init__(self, search_space: Dict[str, Any], constraints: Dict[str, float],
                 dataset_name: str, num_gpus: int = 4, device: str = "cuda",
                 exploration_weight: float = 1.414, train_epochs: int = 100,
                 timeout_seconds: int = 7500, checkpoint_dir: str = None):
        """
        初始化并行搜索
        
        Args:
            search_space: 搜索空间配置
            constraints: 约束条件（内存等）
            dataset_name: 数据集名称（如 'MMAct', 'cifar10'）
            num_gpus: GPU数量
            device: 设备类型
            exploration_weight: 探索权重
            train_epochs: 训练轮数
            timeout_seconds: 超时时间（秒）
            checkpoint_dir: 断点保存目录
        """
        self.search_space = search_space
        self.constraints = constraints
        self.dataset_name = dataset_name
        self.num_gpus = num_gpus
        self.device = device
        self.exploration_weight = exploration_weight
        self.train_epochs = train_epochs
        self.timeout_seconds = timeout_seconds
        
        # 断点恢复目录
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = f"sfs_checkpoints/{timestamp}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # ✅ 获取数据集信息（自动推断任务类型）
        self.dataset_info = get_dataset_info(dataset_name)
        self.task_type = self.dataset_info.get('task_type', 'har')  # 提取任务类型
        
        # 内存约束
        self.max_memory = float(constraints.get("max_peak_memory", 20))
        
        # ✅ 配置生成器（传递task_type）
        self.config_generator = ConfigGenerator(
            search_space, self.dataset_info, self.max_memory,
            task_type=self.task_type
        )
        self.scattering_generator = ScatteringGenerator(self.config_generator)
        self.llm_config_generator = LLMConfigGenerator(
            search_space, constraints, dataset_name
        )
        
        # ✅ 降级策略管理器
        self.degradation_manager = ConfigDegradationManager(
            dataset_info=self.dataset_info,
            max_memory=self.max_memory,
            check_memory_fn=self._check_memory_constraint,
            is_duplicate_fn=self._is_duplicate_config
        )
        
        # 搜索树
        self.tree = MCTSTree(exploration_weight=exploration_weight)
        self.best_candidate: Optional[CandidateModel] = None
        self.best_reward: float = -float('inf')
        self.iteration_count: int = 0
        
        # 量化方向
        self.quant_directions = ["none", "static", "qat", "qaft"]
        
        # 去重
        self.seen_configs = set()
        self.duplicate_count = 0
        self.max_retry_attempts = 3
        
        # 多进程组件
        self.task_queue = None
        self.result_queue = None
        self.workers = []
        
        # 中断标志
        self.interrupted = False
        self.timeout_triggered = False
        
        # 配置生成失败计数
        self.consecutive_generation_failures = 0
        self.max_consecutive_failures = 10
        
        # ✅ 对话记录存储
        self.dialogue_records = []
        self.current_dialogue_record = None
        
        print(f"✅ 并行搜索初始化")
        print(f"   - 数据集: {dataset_name}")
        print(f"   - 任务类型: {self.task_type.upper()}")
        print(f"   - GPU数量: {num_gpus}")
        print(f"   - 量化模式: {', '.join(self.quant_directions)}")
        print(f"   - 训练轮数: {train_epochs}")
        print(f"   - 超时设置: {timeout_seconds}秒")
        print(f"   - 断点目录: {self.checkpoint_dir}")
    
    # ==================== 断点恢复相关方法 ====================
    
    def save_checkpoint(self, filepath: str = None):
        """保存断点"""
        if filepath is None:
            filepath = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
        
        checkpoint = {
            'iteration_count': self.iteration_count,
            'best_reward': self.best_reward,
            'best_candidate': self.best_candidate,
            'seen_configs': self.seen_configs,
            'duplicate_count': self.duplicate_count,
            'tree_state': self._serialize_tree(),
            'timestamp': datetime.now().isoformat(),
            'consecutive_generation_failures': self.consecutive_generation_failures,
            'task_type': self.task_type,  # ✅ 保存任务类型
            'dataset_name': self.dataset_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 断点已保存: {filepath}")
        return filepath
    
    def load_checkpoint(self, filepath: str):
        """加载断点"""
        if not os.path.exists(filepath):
            print(f"⚠️  断点文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.iteration_count = checkpoint['iteration_count']
            self.best_reward = checkpoint['best_reward']
            self.best_candidate = checkpoint['best_candidate']
            self.seen_configs = checkpoint['seen_configs']
            self.duplicate_count = checkpoint['duplicate_count']
            self.consecutive_generation_failures = checkpoint.get(
                'consecutive_generation_failures', 0
            )
            
            self._deserialize_tree(checkpoint['tree_state'])
            
            print(f"✅ 断点已加载: {filepath}")
            print(f"   - 迭代数: {self.iteration_count}")
            print(f"   - 最佳奖励: {self.best_reward:.4f}")
            print(f"   - 已见配置: {len(self.seen_configs)}")
            print(f"   - 连续失败: {self.consecutive_generation_failures}")
            
            return True
        
        except Exception as e:
            print(f"❌ 加载断点失败: {e}")
            return False
    
    def _serialize_tree(self) -> Dict:
        """序列化搜索树"""
        return {
            'nodes': {
                node_id: node.to_dict()
                for node_id, node in self.tree.nodes.items()
            },
            'forest_roots': [root.node_id for root in self.tree.forest_roots],
            'global_insights': self.tree.global_insights
        }
    
    def _deserialize_tree(self, tree_state: Dict):
        """反序列化搜索树"""
        # 重建节点
        for node_id, node_data in tree_state['nodes'].items():
            candidate = CandidateModel(
                config=node_data['candidate_config'],
                task_type=self.task_type  # ✅ 传递任务类型
            )
            candidate.metrics = node_data['metrics']
            candidate.candidate_id = node_id
            
            node = MCTSNode(
                node_id=node_id,
                candidate=candidate,
                directions=list(node_data['direction_stats'].keys())
            )
            node.visit_count = node_data['visit_count']
            node.total_reward = node_data['average_reward'] * node_data['visit_count']
            
            for direction, stats in node_data['direction_stats'].items():
                node.direction_q_values[direction] = stats['q_value']
                node.direction_visits[direction] = stats['visits']
            
            self.tree.nodes[node_id] = node
        
        # 重建森林根节点
        for root_id in tree_state['forest_roots']:
            if root_id in self.tree.nodes:
                root = self.tree.nodes[root_id]
                root.is_forest_root = True
                self.tree.forest_roots.append(root)
        
        # 恢复全局洞察
        self.tree.global_insights = tree_state['global_insights']
        
        # 重建父子关系
        for node_id, node_data in tree_state['nodes'].items():
            node = self.tree.nodes[node_id]
            for direction, child_ids in node_data['children_directions'].items():
                for child_id in child_ids:
                    if child_id in self.tree.nodes:
                        node.add_child(direction, self.tree.nodes[child_id])
    
    # ==================== 工作进程管理 ====================
    
    def _start_workers(self):
        """启动工作进程 - 支持GPU复用"""
        actual_gpus = torch.cuda.device_count()
        print(f"🔍 检测到 {actual_gpus} 个物理GPU")
        sys.stdout.flush()
        print(f"📊 将启动 {self.num_gpus} 个工作进程")
        sys.stdout.flush()
        
        if self.num_gpus > actual_gpus:
            print(f"⚙️  GPU复用: {self.num_gpus} 个进程 → {actual_gpus} 个GPU")
            sys.stdout.flush()
        
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        # 创建队列
        self.task_queue = mp.Queue(maxsize=self.num_gpus * 2)
        self.result_queue = mp.Queue()
        
        # 启动工作进程
        for worker_id in range(self.num_gpus):
            gpu_id = worker_id % actual_gpus  # GPU复用
            worker = mp.Process(
                target=worker_process,
                args=(worker_id, gpu_id, self.task_queue, self.result_queue,
                      self.constraints, self.dataset_name, self.train_epochs)
            )
            worker.start()
            self.workers.append(worker)
            print(f"✅ 启动 Worker-{worker_id} → GPU{gpu_id} (PID: {worker.pid})")
    
    def _stop_workers(self):
        """停止工作进程"""
        print("\n🛑 停止工作进程...")
        
        # 发送终止信号
        for _ in range(self.num_gpus):
            self.task_queue.put(None)
        
        # 等待进程结束
        for i, worker in enumerate(self.workers):
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"⚠️  Worker-{i} 未响应，强制终止")
                worker.terminate()
                worker.join()
        
        print("✅ 所有工作进程已停止")
    
    # ==================== 配置管理 ====================
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """生成配置哈希"""
        normalized = self._normalize_config(config)
        config_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """规范化配置"""
        normalized = {
            "input_channels": config.get("input_channels"),
            "num_classes": config.get("num_classes"),
            "stages": []
        }
        
        for stage in config.get("stages", []):
            normalized_stage = {
                "channels": stage.get("channels"),
                "blocks": []
            }
            
            for block in stage.get("blocks", []):
                normalized_block = {
                    "type": block.get("type"),
                    "kernel_size": block.get("kernel_size"),
                    "stride": block.get("stride"),
                    "expansion": block.get("expansion"),
                    "has_se": block.get("has_se"),
                    "se_ratio": block.get("se_ratio"),
                    "skip_connection": block.get("skip_connection"),
                    "activation": block.get("activation")
                }
                normalized_stage["blocks"].append(normalized_block)
            
            normalized["stages"].append(normalized_stage)
        
        return normalized
    
    def _is_duplicate_config(self, config: Dict[str, Any]) -> bool:
        """检查重复"""
        config_hash = self._generate_config_hash(config)
        return config_hash in self.seen_configs
    
    def _add_config_to_seen(self, config: Dict[str, Any]):
        """添加到已见集合"""
        config_hash = self._generate_config_hash(config)
        self.seen_configs.add(config_hash)
    
    # ==================== 森林初始化 ====================
    
    def initialize_forest(self, num_seeds: int = 5):
        """初始化森林 - 使用工作进程池"""
        print(f"\n🌱 初始化森林: {num_seeds} 个种子")
        sys.stdout.flush()
        
        # 启动工作进程
        self._start_workers()
        
        # 批量生成唯一种子
        unique_seed_configs = []
        max_batches = 8
        batch_count = 0
        
        print(f"\n📋 批量生成唯一种子配置...")
        sys.stdout.flush()
        
        while len(unique_seed_configs) < num_seeds and batch_count < max_batches:
            batch_count += 1
            print(f"\n🔄 第 {batch_count} 批生成 "
                  f"(当前已有 {len(unique_seed_configs)}/{num_seeds} 个唯一种子)")
            sys.stdout.flush()
            
            scattered_seeds = self.scattering_generator.generate_scattered_seeds(num_seeds)
            
            added_in_this_batch = 0
            for i, seed_config in enumerate(scattered_seeds):
                if len(unique_seed_configs) >= num_seeds:
                    print(f"  ✅ 已收集足够种子，停止添加")
                    sys.stdout.flush()
                    break
                
                if not self._is_duplicate_config(seed_config):
                    unique_seed_configs.append(seed_config)
                    self._add_config_to_seen(seed_config)
                    added_in_this_batch += 1
                    print(f"  ✅ 接受种子 {len(unique_seed_configs)}/{num_seeds} "
                          f"(批次内第{i}个)")
                    sys.stdout.flush()
                else:
                    print(f"  🔁 跳过重复种子 (批次内第{i}个)")
                    sys.stdout.flush()
            
            print(f"  📊 本批次添加了 {added_in_this_batch} 个唯一种子")
            sys.stdout.flush()
        
        # 提交所有种子到任务队列
        seed_ids = []
        print(f"\n📤 提交 {len(unique_seed_configs)} 个种子到训练队列:")
        sys.stdout.flush()
        
        for i, seed_config in enumerate(unique_seed_configs):
            seed_id = f"seed_{i}"
            seed_ids.append((seed_id, seed_config, i))
            self.task_queue.put((seed_id, seed_config))
            
            print(f"  Seed {i}:")
            sys.stdout.flush()
            print(f"seed config: {json.dumps(seed_config, indent=4)}")
            sys.stdout.flush()
        
        # 收集所有结果
        print(f"\n⏳ 等待 {len(seed_ids)} 个种子评估完成...")
        sys.stdout.flush()
        
        for seed_id, seed_config, i in seed_ids:
            candidate_id, reward, metrics = self.result_queue.get()
            
            # ✅ 创建候选模型（传递task_type）
            candidate = CandidateModel(
                config=seed_config,
                task_type=self.task_type
            )
            candidate.candidate_id = candidate_id
            candidate.metrics = metrics
            
            # 创建节点
            node = MCTSNode(
                node_id=seed_id,
                candidate=candidate,
                directions=self.quant_directions
            )
            node.update_reward(reward)
            node.visit_count = 0
            node.is_forest_root = True
            
            # 添加到森林
            self.tree.add_node(node, is_forest_root=True)
            
            # 初始化方向
            scattering_directions = self.tree.scattering(node)
            node.directions = scattering_directions
            for direction in scattering_directions:
                node.direction_q_values[direction] = 0.0
                node.direction_visits[direction] = 0
            
            # 更新最佳
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_candidate = candidate
            
            print(f"✅ 种子 {seed_id}: 奖励={reward:.2f}, GPU={metrics.get('gpu_id')}")
            sys.stdout.flush()
    
    # ==================== 搜索主循环 ====================
    
    def search(self, iterations: int = 100, exploration_weight: float = 1.0, 
               max_runtime_hours: float = None):
        """
        执行搜索 - 支持超时检测和中断处理
        
        Args:
            iterations: 迭代次数
            exploration_weight: 探索权重
            max_runtime_hours: 最大运行时长（小时），如果为 None 则不限制
        
        Returns:
            搜索结果字典（包含完成状态）
        """
        print(f"\n🔍 开始搜索: {iterations} 次迭代")
        print(f"⏱️  超时设置: {self.timeout_seconds}秒无新结果将触发保存退出")
        
        # ✅ 新增：记录搜索开始时间
        search_start_time = time.time()
        max_runtime_seconds = max_runtime_hours * 3600 if max_runtime_hours else None
        if max_runtime_seconds:
            print(f"⏰ 最大运行时长: {max_runtime_hours}小时 ({max_runtime_seconds}秒)")
    
        last_result_time = time.time()
        consecutive_empty_count = 0
        
        pending_tasks = {}
        submitted_count = 0
        completed_count = 0
        
        try:
            # ===== 阶段1: 填充GPU任务队列 =====
            print(f"\n📤 阶段1: 填充GPU任务队列")
            sys.stdout.flush()
            max_initial_retries = 10
            
            while submitted_count < min(self.num_gpus, iterations) and \
                  max_initial_retries > 0:
                selected_seed = self.tree.select_forest_root()
                selected_seed.visit_count += 1
                print(f"\n (●￣(ｴ)￣●) selected_seed: {selected_seed.node_id}")
                sys.stdout.flush()
                
                current_node, trajectory = self._simulate_from_seed(selected_seed)
                print(f"current_node: {current_node.node_id}")
                sys.stdout.flush()
                
                direction = current_node.get_best_direction(self.exploration_weight)
                print("direction:", direction)
                sys.stdout.flush()
                
                new_config = self._generate_config_with_llm(current_node, direction)
                
                if self._is_duplicate_config(new_config):
                    print(f"🔁 重复配置，重新生成 "
                          f"(已提交: {submitted_count}/{self.num_gpus})")
                    sys.stdout.flush()
                    max_initial_retries -= 1
                    continue
                
                candidate_id = f"iter_{self.iteration_count}_dir_{direction}"
                self.iteration_count += 1
                submitted_count += 1
                
                self.task_queue.put((candidate_id, new_config))
                self._add_config_to_seen(new_config)
                
                pending_tasks[candidate_id] = (current_node, direction, 
                                               trajectory, new_config)
                print(f"📤 提交任务 {candidate_id} "
                      f"({submitted_count}/{min(self.num_gpus, iterations)})")
                sys.stdout.flush()
            
            # ===== 阶段2: 流水线执行 =====
            print(f"\n🔄 阶段2: 流水线执行")
            sys.stdout.flush()
            
            while completed_count < iterations:
                # ✅ 新增：检查是否超过最大运行时长
                if max_runtime_seconds:
                    elapsed_time = time.time() - search_start_time
                    if elapsed_time >= max_runtime_seconds:
                        print(f"\n⏰ 达到最大运行时长 {max_runtime_hours}小时 "
                            f"({elapsed_time/3600:.2f}h)")
                        print(f"   已完成: {completed_count}/{iterations} 次迭代")
                        self.timeout_triggered = True
                        break

                if self.interrupted:
                    print(f"\n⚠️  检测到中断标志，准备保存并退出...")
                    sys.stdout.flush()
                    break
                
                try:
                    result_id, reward, metrics = self.result_queue.get(timeout=1)
                    print(f"✅ 收到结果 {result_id} "
                          f"(完成: {completed_count+1}/{iterations})")
                    sys.stdout.flush()
                    
                    # 重置超时计数
                    last_result_time = time.time()
                    consecutive_empty_count = 0
                    
                    # 处理结果
                    if result_id in pending_tasks:
                        current_node, direction, trajectory, new_config = \
                            pending_tasks.pop(result_id)
                        
                        # ✅ 创建子节点（传递task_type）
                        new_candidate = CandidateModel(
                            config=new_config,
                            task_type=self.task_type
                        )
                        new_candidate.candidate_id = result_id
                        new_candidate.metrics = metrics
                        new_candidate.iteration = completed_count
                        new_candidate.parent_id = current_node.node_id
                        new_candidate.parent_direction = direction
                        new_candidate.root_seed_id = self._get_root_seed_id(current_node)
                        
                        child_node = MCTSNode(
                            node_id=result_id,
                            candidate=new_candidate,
                            directions=current_node.directions.copy()
                        )
                        child_node.iteration = completed_count
                        child_node.update_reward(reward)
                        
                        # 更新父节点的方向Q值
                        new_leafedge_q = max(
                            current_node.direction_q_values[direction], reward
                        )
                        visits = current_node.direction_visits.get(direction, 0)
                        alpha = 1.0 / (visits + 1) if visits > 0 else 1.0
                        leafedge_q = (1 - alpha) * \
                            current_node.direction_q_values[direction] + \
                            alpha * new_leafedge_q
                        
                        print(f"[DEBUG] 更新节点 {current_node.node_id} "
                              f"方向 {direction} 的 Q 值: {leafedge_q:.4f} "
                              f"(原始: {current_node.direction_q_values[direction]:.4f})")
                        sys.stdout.flush()
                        
                        current_node.direction_q_values[direction] = leafedge_q
                        
                        # 添加到树
                        self.tree.add_node(child_node, current_node, direction)
                        
                        # 反向传播
                        self._backpropagate(
                            trajectory + [(current_node, direction, child_node)],
                            reward
                        )
                        
                        # Scouting
                        feedback = {
                            "reward": reward,
                            "accuracy": metrics.get('accuracy', 0),
                            "direction": direction,
                            "parent_config": current_node.candidate.config,
                            "child_config": new_config
                        }
                        self.tree.scouting(current_node, direction, child_node, 
                                          reward, feedback)
                        
                        # 更新最佳
                        if reward > self.best_reward:
                            self.best_reward = reward
                            self.best_candidate = new_candidate
                            print(f"🎯 新最佳! 奖励={reward:.2f}")
                            sys.stdout.flush()
                        
                        completed_count += 1
                        
                        # 每10个任务打印一次进度
                        if completed_count % 10 == 0:
                            self._print_search_progress(completed_count)
                            checkpoint_path = self.save_checkpoint()
                            print(f"💾 自动保存断点: {checkpoint_path}")
                    
                    # 补充新任务
                    if self.iteration_count < iterations and not self.interrupted:
                        retry_count = 0
                        max_retries = 5
                        
                        while retry_count < max_retries:
                            selected_seed = self.tree.select_forest_root()
                            print(f"ROOT selected_seed for new task: "
                                  f"{selected_seed.node_id}")
                            sys.stdout.flush()
                            selected_seed.visit_count += 1
                            
                            current_node, trajectory = \
                                self._simulate_from_seed(selected_seed)
                            direction = current_node.get_best_direction(
                                self.exploration_weight
                            )
                            
                            new_config = self._generate_config_with_llm(
                                current_node, direction
                            )
                            
                            if not self._is_duplicate_config(new_config):
                                candidate_id = \
                                    f"iter_{self.iteration_count}_dir_{direction}"
                                self.iteration_count += 1
                                
                                self.task_queue.put((candidate_id, new_config))
                                self._add_config_to_seen(new_config)
                                
                                pending_tasks[candidate_id] = (
                                    current_node, direction, trajectory, new_config
                                )
                                print(f"📤 补充任务 {candidate_id} "
                                      f"(待完成: {iterations - completed_count})")
                                sys.stdout.flush()
                                break
                            else:
                                retry_count += 1
                                self.duplicate_count += 1
                                print(f"🔁 重复配置，重试 {retry_count}/{max_retries}")
                                sys.stdout.flush()
                        
                        if retry_count == max_retries:
                            print(f"⚠️  多次重试仍重复，跳过本次补充")
                            sys.stdout.flush()
                
                except Empty:
                    # ✅ 在超时检查前也检查运行时长
                    if max_runtime_seconds:
                        elapsed_time = time.time() - search_start_time
                        if elapsed_time >= max_runtime_seconds:
                            print(f"\n⏰ 达到最大运行时长 {max_runtime_hours}小时")
                            self.timeout_triggered = True
                            break
                        
                    consecutive_empty_count += 1
                    time_since_last_result = time.time() - last_result_time
                    
                    if time_since_last_result > self.timeout_seconds:
                        print(f"\n⚠️  超时检测: 已 {time_since_last_result:.0f}秒 无新结果")
                        sys.stdout.flush()
                        print(f"   待处理任务: {len(pending_tasks)}")
                        sys.stdout.flush()
                        print(f"   已完成: {completed_count}/{iterations}")
                        sys.stdout.flush()
                        
                        if len(pending_tasks) == 0:
                            print(f"\n🚨 检测到异常: 无待处理任务且长时间无结果")
                            sys.stdout.flush()
                            self.timeout_triggered = True
                            break
                        
                        if time_since_last_result > self.timeout_seconds * 2:
                            print(f"\n🚨 严重超时: {time_since_last_result:.0f}秒无响应")
                            sys.stdout.flush()
                            print(f"   强制保存当前状态并退出")
                            sys.stdout.flush()
                            self.timeout_triggered = True
                            break
                    
                    if consecutive_empty_count <= 5 or \
                       consecutive_empty_count % 30 == 0:
                        print(f"⏳ 等待结果... (待处理: {len(pending_tasks)}, "
                              f"已等待: {time_since_last_result:.0f}秒)")
                        sys.stdout.flush()
                    
                    continue
            
            # 输出最终状态
            if self.interrupted:
                print(f"\n⚠️  用户中断 - 已完成 {completed_count}/{iterations} 次迭代")
                sys.stdout.flush()
            elif self.timeout_triggered:
                print(f"\n⚠️  超时退出 - 已完成 {completed_count}/{iterations} 次迭代")
                sys.stdout.flush()
            else:
                print(f"\n✅ 搜索完成！共完成 {completed_count} 次迭代")
                sys.stdout.flush()
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  检测到 KeyboardInterrupt")
            sys.stdout.flush()
            print(f"   已完成: {completed_count}/{iterations}")
            sys.stdout.flush()
            print(f"   待处理任务: {len(pending_tasks)}")
            sys.stdout.flush()
            self.interrupted = True
        
        except Exception as e:
            print(f"\n❌ 搜索过程出现异常: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            self.interrupted = True
        
        finally:
            print(f"\n🧹 清理资源并保存状态...")
            sys.stdout.flush()
            self._stop_workers()
            
            return {
                'completed_iterations': completed_count,
                'total_iterations': iterations,
                'interrupted': self.interrupted,
                'timeout_triggered': self.timeout_triggered,
                'pending_tasks': len(pending_tasks)
            }
    
    # ==================== MCTS核心方法 ====================
    
    def _simulate_from_seed(self, seed_node: MCTSNode) -> Tuple[MCTSNode, List]:
        """从种子模拟"""
        print(f"\n🔍 从种子 {seed_node.node_id} 模拟")
        sys.stdout.flush()
        
        trajectory = []
        current_node = seed_node
        max_depth = random.randint(7, 40)
        depth = 0
        
        while depth < max_depth and current_node.children:
            direction = current_node.get_best_direction(self.exploration_weight)
            print(f"  ➡️ 选择方向: {direction}")
            sys.stdout.flush()
            
            if direction not in current_node.children or \
               len(current_node.children[direction]) == 0:
                break
            
            next_node = self._select_best_child_from_direction(
                current_node, direction, self.exploration_weight
            )
            if next_node is None:
                break
            
            trajectory.append((current_node, direction, next_node))
            current_node = next_node
            depth += 1
        
        return current_node, trajectory
    
    def _select_best_child_from_direction(
        self, parent_node: MCTSNode,
        direction: str,
        exploration_weight: float
    ) -> Optional[MCTSNode]:
        """从某个方向的多个子节点中选择UCT最佳的"""
        candidates = parent_node.children.get(direction, [])
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # UCT选择
        best_child = None
        best_score = -float('inf')
        total_visits = sum(child.visit_count for child in candidates)
        
        for child in candidates:
            if child.visit_count == 0:
                return child
            
            exploitation = child.node_q_value
            exploration = exploration_weight * np.sqrt(
                np.log(total_visits) / child.visit_count
            )
            uct_score = exploitation + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child
    
    def _backpropagate(self, trajectory: List, reward: float):
        """反向传播"""
        print(f"\n🔄 反向传播奖励: {reward:.4f}")
        sys.stdout.flush()
        print(f"trajectory 长度: {len(trajectory)}\n")
        sys.stdout.flush()
        
        for i, (node, direction, next_node) in enumerate(reversed(trajectory)):
            print(f"反转轨迹{i} node: {node.node_id}, direction: {direction} "
                  f"next_node: {next_node.node_id}")
            sys.stdout.flush()
            
            current_q = node.direction_q_values.get(direction, 0.0)
            print(f"[DEBUG] 当前节点 {node.node_id} 方向 {direction} 的 Q 值: "
                  f"{current_q:.4f}")
            sys.stdout.flush()
            
            # 获取子节点的最大Q值
            next_max_q = 0.0
            if next_node and next_node.direction_q_values:
                next_max_q = max(next_node.direction_q_values.values())
            
            new_q = max(current_q, next_max_q)
            visits = node.direction_visits.get(direction, 0)
            alpha = 1.0 / (visits + 1) if visits > 0 else 1.0
            
            updated_q = (1 - alpha) * current_q + alpha * new_q
            print(f"[DEBUG] 更新节点 {node.node_id} 方向 {direction} 的 Q 值: "
                  f"{updated_q:.4f} (当前: {current_q:.4f}, 新 Q: {new_q:.4f})")
            sys.stdout.flush()
            
            node.direction_q_values[direction] = updated_q
            node.direction_visits[direction] += 1
            print(f"[DEBUG] 节点 {node.node_id} 方向 {direction} 的访问次数: "
                  f"{node.direction_visits[direction]}")
            sys.stdout.flush()
    
    def _get_root_seed_id(self, node: MCTSNode) -> str:
        """追溯到根种子"""
        current = node
        while not current.is_forest_root:
            parent_found = False
            for potential_parent in self.tree.nodes.values():
                for direction, children_list in potential_parent.children.items():
                    if any(c.node_id == current.node_id for c in children_list):
                        current = potential_parent
                        parent_found = True
                        break
                if parent_found:
                    break
            if not parent_found:
                break
        return current.node_id
    
    # ==================== 配置生成 ====================
    
    def _check_memory_constraint(self, config: Dict[str, Any]) -> \
            Tuple[bool, float, str]:
        """
        检查配置的内存使用情况
        ✅ 自动适配HAR和Image任务
        """
        try:
            candidate = CandidateModel(
                config=config,
                task_type=self.task_type  # ✅ 传递任务类型
            )
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
            
            # 计算内存使用量
            memory_info = calculate_memory_usage(
                model,
                input_size=input_size,
                device='cpu'
            )
            
            memory_usage = memory_info["total_memory_MB"]
            
            # 根据量化模式调整内存使用量
            quant_mode = config.get("quant_mode", "none")
            if quant_mode in ["static", "qat", "qaft"]:
                compressed_memory = memory_usage / 4.0
                print(f"📦 量化压缩: {memory_usage:.2f}MB → "
                      f"{compressed_memory:.2f}MB (模式: {quant_mode})")
                memory_usage = compressed_memory
            
            # 检查是否超过限制
            if memory_usage <= self.max_memory:
                return True, memory_usage, "OK"
            else:
                error_msg = (f"内存使用 {memory_usage:.2f}MB "
                            f"超过限制 {self.max_memory}MB")
                return False, memory_usage, error_msg
        
        except Exception as e:
            print(f"⚠️ 内存计算失败: {e}")
            return False, 0, f"内存计算失败: {str(e)}"
    
    def _generate_config_with_llm(self, parent_node: MCTSNode, 
                                   direction: str) -> Dict[str, Any]:
        """
        使用LLM生成配置 - 增强版（记录完整对话）
        包含内存检查和重试机制
        """
        parent_config = parent_node.candidate.config if parent_node.candidate else {}
        parent_performance = {
            "average_reward": parent_node.average_reward,
            "visit_count": parent_node.visit_count,
            "directions_explored": list(parent_node.direction_visits.keys())
        }
        global_insights = self.tree.global_insights
        
        # ✅ 初始化对话记录
        dialogue_record = self.llm_config_generator.init_dialogue_record(
            parent_node, direction
        )
        self.current_dialogue_record = dialogue_record
        
        memory_feedback = None
        llm_failed = False
        measured_memory = None
        
        # 检查是否需要紧急降级
        emergency_mode = self.consecutive_generation_failures >= \
                        self.max_consecutive_failures
        if emergency_mode:
            print(f"🚨 进入紧急降级模式 "
                  f"(连续失败 {self.consecutive_generation_failures} 次)")
            sys.stdout.flush()
        
        for attempt in range(self.max_retry_attempts + 1):
            try:
                # 如果LLM失败，直接使用降级配置
                if emergency_mode or (attempt >= 2 and 
                                     (llm_failed or memory_feedback is not None)):
                    print(f"⚠️ LLM 已失败，使用降级配置 "
                          f"(尝试 {attempt+1}/{self.max_retry_attempts+1})")
                    sys.stdout.flush()
                    
                    estimated_memory = measured_memory if measured_memory \
                                      else self.max_memory * 1.5
                    print(f"   - 估算内存: {estimated_memory:.2f}MB")
                    sys.stdout.flush()
                    
                    dup_modification = (estimated_memory <= self.max_memory)
                    
                    new_config = self.degradation_manager.generate_degraded_config(
                        parent_config, direction, estimated_memory, 
                        dup_modification, dialogue_record
                    )
                else:
                    try:
                        # 调用LLM生成配置
                        new_config = self.llm_config_generator.\
                            generate_config_with_context(
                                dialogue_record,
                                parent_config, direction, parent_performance,
                                global_insights, memory_feedback=memory_feedback
                            )
                    
                    except Exception as llm_e:
                        print(f"❌ LLM调用失败: {llm_e}")
                        llm_failed = True
                        estimated_memory = measured_memory if measured_memory \
                                          else self.max_memory * 1.5
                        dup_modification = (estimated_memory <= self.max_memory)
                        new_config = self.degradation_manager.\
                            generate_degraded_config(
                                parent_config, direction, estimated_memory,
                                dup_modification, dialogue_record
                            )
                
                # 检查内存
                print(f"🔧 生成配置 (尝试 {attempt+1}/{self.max_retry_attempts+1})")
                
                memory_ok, memory_usage, memory_msg = \
                    self._check_memory_constraint(new_config)
                measured_memory = memory_usage
                print(f"📊 内存使用: {memory_usage:.2f}MB "
                      f"(限制: {self.max_memory}MB)")
                
                # 检查重复
                if self._is_duplicate_config(new_config):
                    print(f"🔁 LLM生成重复配置 "
                          f"(尝试 {attempt+1}/{self.max_retry_attempts+1})")
                    
                    # ✅ 记录重复反馈
                    memory_feedback = f"""
                    --- FEEDBACK {attempt+1} (MCTS Evaluation) ---
                    **FAILURE: DUPLICATE CONFIGURATION**
                    The generated configuration is identical to a previously evaluated model.

                    **Current Config Hash:** {self._generate_config_hash(new_config)[:16]}...
                    **Attempt:** {attempt+1}/{self.max_retry_attempts+1}

                    **ACTION REQUIRED:** Generate a DIFFERENT architecture by:
                    - Changing the number of stages (currently: {len(new_config.get('stages', []))})
                    - Using different conv types (MBConv, DWSepConv, SeSepConv, DpConv, SeDpConv)
                    - Adjusting channel numbers significantly
                    - Adjust channel numbers significantly
                    - Try different expansion ratios
                    - Modify SE module settings
                    """
                    dialogue_record["messages"].append({
                        "role": "user",
                        "content": memory_feedback
                    })
                    
                    if attempt == self.max_retry_attempts:
                        print("🚨 达到最大重试次数，使用强制降级配置")
                        fallback_memory = memory_usage if memory_usage > 0 \
                                         else self.max_memory * 1.1
                        dup_modification = (fallback_memory <= self.max_memory)
                        final_config = self.degradation_manager.\
                            generate_degraded_config(
                                parent_config, direction, fallback_memory,
                                dup_modification, dialogue_record
                            )
                        self._finalize_dialogue_record(dialogue_record, 
                                                       final_config, success=False)
                        return final_config
                    
                    continue
                
                # 检查内存是否通过
                if memory_ok:
                    print(f"✅ 配置通过所有检查: {memory_usage:.2f}MB")
                    self.consecutive_generation_failures = 0
                    
                    # ✅ 记录成功
                    success_msg = f"""
                    --- EVALUATION COMPLETE ---
                    **SUCCESS:** Configuration passed all checks.
                    - Memory: {memory_usage:.2f} MB (Limit: {self.max_memory} MB)
                    - Status: ACCEPTED

                    **FINAL CONFIGURATION CONFIRMED.**
                    """
                    dialogue_record["messages"].append({
                        "role": "user",
                        "content": success_msg
                    })
                    
                    self._finalize_dialogue_record(dialogue_record, 
                                                   new_config, success=True)
                    return new_config
                else:
                    print(f"⚠️ 配置内存超标 "
                          f"({attempt+1}/{self.max_retry_attempts+1}): {memory_msg}")
                    
                    if attempt == self.max_retry_attempts:
                        print("🚨 达到最大重试次数，使用降级配置")
                        self.consecutive_generation_failures = 0
                        dup_modification = (memory_usage <= self.max_memory)
                        final_config = self.degradation_manager.\
                            generate_degraded_config(
                                parent_config, direction, memory_usage,
                                dup_modification, dialogue_record
                            )
                        self._finalize_dialogue_record(dialogue_record, 
                                                       final_config, success=False)
                        return final_config
                    
                    # 改进后的反馈
                    reduction_needed = memory_usage / self.max_memory
                    specific_suggestions = []
                    
                    if reduction_needed > 1.5:
                        specific_suggestions.append(
                            "CRITICAL: Memory is 50%+ over limit. "
                            "Reduce stages to 2-3 maximum"
                        )
                        specific_suggestions.append(
                            f"Set all channel numbers to 8-16 range"
                        )
                        specific_suggestions.append(
                            f"Use bigger strides (e.g., 2 or 4) in early layers"
                        )
                    elif reduction_needed > 1.2:
                        specific_suggestions.append(
                            f"Reduce channel numbers by "
                            f"~{int((reduction_needed-1)*100)}%"
                        )
                        specific_suggestions.append(
                            "Remove 1-2 blocks from each stage"
                        )
                    elif reduction_needed > 1.0:
                        specific_suggestions.append(
                            "Slightly reduce channel numbers "
                            "(e.g., 16→14, 32→28)"
                        )
                        specific_suggestions.append(
                            "Reduce kernel size in one layer (7→5 or 5→3)"
                        )
                        specific_suggestions.append(
                            "Reduce expansion ratio (e.g., 6→4)"
                        )
                    
                    if attempt > 1:
                        specific_suggestions.append(
                            "⚠️ IMPORTANT: Previous attempts generated duplicates. Try:"
                        )
                        specific_suggestions.append(
                            "  - Use different conv types "
                            "(e.g., DpConv instead of SeDpConv)"
                        )
                        specific_suggestions.append(
                            "  - Use unusual channel numbers "
                            "(e.g., 11, 13, 19 instead of 8, 16)"
                        )
                        specific_suggestions.append(
                            "  - Vary expansion ratios (2, 3, 5 instead of common 4, 6)"
                        )
                    
                    memory_feedback = f"""
                    Previous config memory: {memory_usage:.2f}MB (limit: {self.max_memory}MB)
                    Over budget by: {(reduction_needed-1)*100:.0f}%

                    SPECIFIC ACTIONS REQUIRED:
                    {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(specific_suggestions))}

                    Current config:
                    {json.dumps(new_config, indent=2)}
                    """
                    dialogue_record["messages"].append({
                        "role": "user",
                        "content": memory_feedback
                    })
            
            except Exception as e:
                print(f"❌ LLM配置生成失败 (尝试 {attempt+1}): {e}")
                import traceback
                traceback.print_exc()
                
                dialogue_record["messages"].append({
                    "role": "user",
                    "content": (f"--- EXCEPTION ---\n**ERROR:** {str(e)}\n"
                               f"**Traceback:** {traceback.format_exc()}")
                })
                
                if attempt == self.max_retry_attempts:
                    fallback_memory = measured_memory if measured_memory and \
                                     measured_memory > 0 else self.max_memory * 1.5
                    dup_modification = (fallback_memory <= self.max_memory)
                    final_config = self.degradation_manager.\
                        generate_degraded_config(
                            parent_config, direction, fallback_memory,
                            dup_modification, dialogue_record
                        )
                    self._finalize_dialogue_record(dialogue_record, 
                                                   final_config, success=False)
                    return final_config
        
        # 最终回退
        print("🚨 所有尝试失败，使用降级配置")
        final_memory = measured_memory if measured_memory and \
                      measured_memory > 0 else self.max_memory * 1.5
        self.consecutive_generation_failures += 1
        dup_modification = (final_memory <= self.max_memory)
        final_config = self.degradation_manager.generate_degraded_config(
            parent_config, direction, final_memory, dup_modification,
            dialogue_record
        )
        self._finalize_dialogue_record(dialogue_record, final_config, success=False)
        return final_config
    
    def _finalize_dialogue_record(self, dialogue_record: Dict[str, Any],
                                  final_config: Dict[str, Any],
                                  success: bool):
        """完成对话记录并保存"""
        dialogue_record["final_config"] = final_config
        dialogue_record["success"] = success
        dialogue_record["total_attempts"] = len([
            m for m in dialogue_record["messages"]
            if m["role"] == "assistant"
        ])
        dialogue_record["timestamp"] = datetime.now().isoformat()
        
        self.dialogue_records.append(dialogue_record)
        
        print(f"💾 对话记录已保存: {dialogue_record['dialogue_id']}")
    
    def save_dialogue_records(self, output_dir: str):
        """保存所有对话记录到文件"""
        if not self.dialogue_records:
            print("⚠️ 没有对话记录可保存")
            return
        
        dialogue_file = os.path.join(output_dir, "dialogue_records.json")
        
        try:
            with open(dialogue_file, 'w', encoding='utf-8') as f:
                json.dump(self.dialogue_records, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 对话记录已保存: {dialogue_file}")
            print(f"   总对话数: {len(self.dialogue_records)}")
            
            success_count = sum(1 for r in self.dialogue_records 
                              if r.get('success', False))
            avg_attempts = sum(r.get('total_attempts', 0) 
                             for r in self.dialogue_records) / \
                          len(self.dialogue_records)
            
            print(f"   成功率: {success_count}/{len(self.dialogue_records)} "
                  f"({success_count/len(self.dialogue_records)*100:.1f}%)")
            print(f"   平均尝试次数: {avg_attempts:.2f}")
        
        except Exception as e:
            print(f"❌ 保存对话记录失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== 工具方法 ====================
    
    def _print_search_progress(self, iteration: int):
        """打印进度"""
        stats = self.tree.get_graph_statistics()
        
        print(f"\n--- 迭代 {iteration} 进度 ---")
        print(f"最佳奖励: {self.best_reward:.4f}")
        print(f"总节点数: {stats['total_nodes']}")
        print(f"评估节点: {stats['evaluated_nodes']}")
        print(f"重复配置: {self.duplicate_count}")
        print("----------------------------\n")
    
    def get_best_model(self) -> Optional[CandidateModel]:
        """获取最佳模型"""
        return self.best_candidate
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.tree.get_graph_statistics()
        return {
            "iterations": self.iteration_count,
            "best_reward": self.best_reward,
            "tree_statistics": stats,
            "global_insights": self.tree.global_insights,
            "unique_configs": len(self.seen_configs),
            "duplicate_count": self.duplicate_count,
            "interrupted": self.interrupted,
            "timeout_triggered": self.timeout_triggered,
            "task_type": self.task_type  # ✅ 添加任务类型
        }
    
    def save_search_state(self, filepath: str):
        """保存搜索状态"""
        self.tree.save_graph_info(filepath)


# ==================== 工厂函数 ====================

def create_parallel_sfs_search(
    search_space: Dict[str, Any],
    constraints: Dict[str, float],
    dataset_name: str = "MMAct",
    num_gpus: int = 4,
    device: str = "cuda",
    exploration_weight: float = 1.414,
    train_epochs: int = 200,
    timeout_seconds: int = 7500,
    checkpoint_dir: str = None
) -> ParallelScatteredForestSearch:
    """
    创建并行SFS搜索实例
    
    Args:
        search_space: 搜索空间配置
        constraints: 约束条件
        dataset_name: 数据集名称（会自动推断任务类型）
        num_gpus: GPU数量
        device: 设备类型
        exploration_weight: 探索权重
        train_epochs: 训练轮数
        timeout_seconds: 超时时间
        checkpoint_dir: 断点目录
    
    Returns:
        ParallelScatteredForestSearch实例
    """
    return ParallelScatteredForestSearch(
        search_space=search_space,
        constraints=constraints,
        dataset_name=dataset_name,
        num_gpus=num_gpus,
        device=device,
        exploration_weight=exploration_weight,
        train_epochs=train_epochs,
        timeout_seconds=timeout_seconds,
        checkpoint_dir=checkpoint_dir
    )