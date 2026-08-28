# /root/unified_nas/mcts/mcts_nodes.py
"""
统一的MCTS节点和树结构
支持HAR和Image任务的神经架构搜索

核心特性:
- 支持多任务类型 (task_type参数)
- Foresting: 多棵搜索树并行
- Scattering: 多样化的量化方向
- Scouting: 跨分支经验共享
- 支持图结构的MCTS搜索树
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
import uuid
import numpy as np
import sys

from models import CandidateModel


@dataclass
class MCTSNode:
    """
    MCTS搜索树节点
    
    属性:
        node_id: 节点唯一标识符
        candidate: 候选模型 (包含task_type)
        parent: 父节点
        parent_direction: 来自父节点的量化方向
        children: 子节点字典 {direction: [nodes]}
        visit_count: 访问次数
        total_reward: 累计奖励
        directions: 可用的量化方向
        direction_q_values: 每个方向的Q值
        direction_visits: 每个方向的访问次数
        graph_children: 图结构中所有子节点的ID集合
        is_forest_root: 是否为森林根节点
    """
    node_id: str
    candidate: CandidateModel
    parent: Optional['MCTSNode'] = None
    parent_direction: Optional[str] = None
    children: Dict[str, List['MCTSNode']] = field(default_factory=dict)
    visit_count: int = 0
    total_reward: float = 0.0
    directions: List[str] = field(default_factory=list)
    direction_q_values: Dict[str, float] = field(default_factory=dict)
    direction_visits: Dict[str, int] = field(default_factory=dict)
    graph_children: Set[str] = field(default_factory=set)
    is_forest_root: bool = False
    
    def __post_init__(self):
        """初始化方向相关的字典"""
        if not self.directions:
            self.directions = ["none", "static", "qat", "qaft"]
        
        for direction in self.directions:
            self.direction_q_values[direction] = 0.0
            self.direction_visits[direction] = 0
    
    @property
    def average_reward(self) -> float:
        """计算平均奖励"""
        return self.total_reward / max(1, self.visit_count)
    
    @property
    def node_q_value(self) -> float:
        """
        节点的Q-value = 所有方向的最大Q-value
        如果所有Q-value都是0，则返回average_reward
        """
        max_q = max(self.direction_q_values.values())
        if max_q == 0.0 and self.visit_count > 0:
            return self.average_reward
        return max_q

    def add_child(self, direction: str, child_node: 'MCTSNode'):
        """
        添加子节点 - 支持同一方向多个子节点
        
        参数:
            direction: 量化方向
            child_node: 子节点
        """
        if direction not in self.children:
            self.children[direction] = []

        self.children[direction].append(child_node)
        self.graph_children.add(child_node.node_id)
        child_node.parent = self
        child_node.parent_direction = direction

    def get_children_in_direction(self, direction: str) -> List['MCTSNode']:
        """获取某个方向的所有子节点"""
        return self.children.get(direction, [])
    
    def update_reward(self, reward: float):
        """更新节点奖励"""
        self.visit_count += 1
        self.total_reward += reward
    
    def update_direction_stats(self, direction: str, reward: float):
        """
        更新方向统计信息
        
        参数:
            direction: 量化方向
            reward: 奖励值
        """
        if direction in self.direction_visits:
            self.direction_visits[direction] += 1
    
    def get_best_direction(self, exploration_weight: float = 1.0) -> str:
        """
        使用UCT公式选择最佳方向
        在当前节点选择下一步要探索的最佳量化方向
        
        参数:
            exploration_weight: 探索权重
        
        返回:
            最佳量化方向
        """
        print(f"[DEBUG] 节点{self.node_id}选择方向:")
        sys.stdout.flush()
        print(f"  - direction_visits: {self.direction_visits}")
        sys.stdout.flush()
        print(f"  - direction_q_values: {self.direction_q_values}")
        sys.stdout.flush()
        
        best_direction = None
        best_score = -float('inf')
        total_visits = sum(self.direction_visits.values())
        
        for direction in self.directions:
            if self.direction_visits[direction] == 0:
                # 未探索的方向优先选择
                print(f"  → 返回未探索方向: {direction}")
                sys.stdout.flush()
                return direction
            
            # UCT公式: exploitation + exploration
            exploitation = self.direction_q_values[direction]
            exploration = exploration_weight * np.sqrt(
                np.log(total_visits) / self.direction_visits[direction]
            )
            uct_score = exploitation + exploration
            
            print(f"  - {direction}: Q={exploitation:.2f}, "
                  f"exploration={exploration:.2f}, UCT={uct_score:.2f}")
            sys.stdout.flush()
            
            if uct_score > best_score:
                best_score = uct_score
                best_direction = direction
        
        print(f"  → 返回最佳UCT方向: {best_direction}")
        sys.stdout.flush()
        return best_direction
    
    def get_graph_info(self) -> Dict[str, Any]:
        """获取节点的图结构信息"""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "parent_direction": self.parent_direction,
            "children_count": len(self.graph_children),
            "children_ids": list(self.graph_children),
            "is_forest_root": self.is_forest_root
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 - 包含图结构信息"""
        return {
            "node_id": self.node_id,
            "candidate_config": self.candidate.config if self.candidate else None,
            "task_type": self.candidate.task_type if self.candidate else None,  # ✅ 添加task_type
            "metrics": self.candidate.metrics if self.candidate else {},
            "visit_count": self.visit_count,
            "average_reward": self.average_reward,
            "children_directions": {
                direction: [child.node_id for child in children_list]
                for direction, children_list in self.children.items()
            },
            "direction_stats": {
                dir: {
                    "q_value": self.direction_q_values[dir],
                    "visits": self.direction_visits[dir]
                } for dir in self.directions
            },
            "graph_info": self.get_graph_info()
        }


class MCTSTree:
    """
    MCTS图结构 - 支持真正的图结构和森林搜索
    
    核心功能:
    - Foresting: 多棵搜索树并行探索
    - Scattering: 基于全局经验的方向优先级调整
    - Scouting: 跨分支经验共享
    
    属性:
        nodes: 所有节点的字典 {node_id: node}
        edges: 边的字典 {parent_id: {direction: [child_nodes]}}
        parent_map: 子节点到父节点的映射 {child_id: parent_id}
        forest_roots: 森林中所有根节点的列表
        exploration_weight: UCT探索权重
        global_insights: 全局经验字典
    """
    
    def __init__(self, exploration_weight: float = 1.414):
        self.nodes: Dict[str, MCTSNode] = {}
        self.edges: Dict[str, Dict[str, List[MCTSNode]]] = {}
        self.parent_map: Dict[str, str] = {}
        self.forest_roots: List[MCTSNode] = []
        self.root: Optional[MCTSNode] = None
        self.exploration_weight = exploration_weight
        self.global_insights: Dict[str, Dict[str, Any]] = {}
    
    def add_node(self, node: MCTSNode, parent: Optional[MCTSNode] = None, 
                 direction: Optional[str] = None, is_forest_root: bool = False):
        """
        添加节点到树中
        
        参数:
            node: 要添加的节点
            parent: 父节点 (可选)
            direction: 量化方向 (可选)
            is_forest_root: 是否为森林根节点
        """
        self.nodes[node.node_id] = node
        
        if is_forest_root:
            node.is_forest_root = True
            self.forest_roots.append(node)

        if parent:
            if parent.node_id not in self.edges:
                self.edges[parent.node_id] = {}
            if direction not in self.edges[parent.node_id]:
                self.edges[parent.node_id][direction] = []
            self.edges[parent.node_id][direction].append(node)
            
            self.parent_map[node.node_id] = parent.node_id
            parent.add_child(direction, node)
    
    def get_node(self, node_id: str) -> Optional[MCTSNode]:
        """根据ID获取节点"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[MCTSNode]:
        """获取指定节点的所有子节点"""
        if node_id not in self.edges:
            return []
        all_children = []
        for children_list in self.edges[node_id].values():
            all_children.extend(children_list)
        return all_children
    
    def get_parent(self, node_id: str) -> Optional[MCTSNode]:
        """获取父节点"""
        parent_id = self.parent_map.get(node_id)
        return self.nodes.get(parent_id) if parent_id else None
    
    def get_forest_roots(self) -> List[MCTSNode]:
        """获取所有森林根节点"""
        return self.forest_roots
    
    def select_forest_root(self) -> MCTSNode:
        """
        使用UCT选择森林中的根节点进行扩展
        在多棵搜索树（森林）中选择要扩展的根节点
        
        返回:
            选中的根节点
        """
        if not self.forest_roots:
            # 如果没有根节点，创建一个
            # 注意: 需要在外部创建合适的CandidateModel
            raise RuntimeError("No forest roots available. Create root nodes externally.")
        
        best_root = None
        best_score = -float('inf')
        
        for root in self.forest_roots:
            if root.visit_count == 0:
                print(f"[DEBUG] 选择未访问的森林根节点: {root.node_id}")
                sys.stdout.flush()
                return root
            
            # 使用Q-value而不是average_reward
            exploitation = root.node_q_value
            total_visits = sum(r.visit_count for r in self.forest_roots)
            exploration = self.exploration_weight * np.sqrt(
                np.log(total_visits) / root.visit_count
            )
            uct_score = exploitation + exploration
            
            print(f"[DEBUG] 森林根节点 {root.node_id} 的 UCT 分数: {uct_score:.2f} "
                  f"(Q={exploitation:.2f}, exp={exploration:.2f}, visits={root.visit_count})")
            sys.stdout.flush()
            
            if uct_score > best_score:
                best_score = uct_score
                best_root = root
        
        return best_root or self.forest_roots[0]

    def scattering(self, node: MCTSNode) -> List[str]:
        """
        Scattering: 生成多样化的量化方向
        基于全局insights调整方向优先级
        
        参数:
            node: 当前节点
        
        返回:
            排序后的方向列表
        """
        base_directions = ["none", "static", "qat", "qaft"]
        
        scored_directions = []
        for direction in base_directions:
            insight_key = f"direction_{direction}"
            if insight_key in self.global_insights:
                insight = self.global_insights[insight_key]
                score = insight.get('success_rate', 0.5)
            else:
                score = 0.5
            scored_directions.append((direction, score))
        
        # 按成功率排序
        scored_directions.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_directions]
    
    def scouting(self, parent: MCTSNode, direction: str, child: MCTSNode, 
                 reward: float, feedback: Dict = None):
        """
        Scouting: 跨分支共享经验
        
        参数:
            parent: 父节点
            direction: 量化方向
            child: 子节点
            reward: 奖励值
            feedback: 额外反馈信息 (包含config等)
        """
        insight_key = f"direction_{direction}"
        
        if insight_key not in self.global_insights:
            self.global_insights[insight_key] = {
                'total_reward': 0.0,
                'visit_count': 0,
                'success_count': 0,
                'recent_rewards': [],
                'successful_configs': []
            }
        
        insight = self.global_insights[insight_key]
        insight['total_reward'] += reward
        insight['visit_count'] += 1
        insight['recent_rewards'].append(reward)

        # 动态阈值判断
        if insight['visit_count'] > 1:
            avg_reward = insight['total_reward'] / insight['visit_count']
            if reward > avg_reward * 0.9:
                insight['success_count'] += 1
                if feedback and 'child_config' in feedback:
                    config_insight = self._extract_config_insight(
                        feedback['parent_config'], 
                        feedback['child_config'], 
                        reward
                    )
                    insight['successful_configs'].append(config_insight)
                    if len(insight['successful_configs']) > 5:
                        insight['successful_configs'].pop(0)
        else:
            if reward > 0.5:
                insight['success_count'] += 1

        # 限制最近奖励列表长度
        if len(insight['recent_rewards']) > 10:
            insight['recent_rewards'].pop(0)
        
        # 计算成功率
        insight['success_rate'] = insight['success_count'] / insight['visit_count']
        insight['average_reward'] = insight['total_reward'] / insight['visit_count']
    
    def _extract_config_insight(self, parent_config: Dict[str, Any], 
                                child_config: Dict[str, Any], 
                                reward: float) -> Dict[str, Any]:
        """
        提取父子配置的关键差异
        
        参数:
            parent_config: 父配置
            child_config: 子配置
            reward: 奖励值
        
        返回:
            差异分析字典
        """
        parent_stages = parent_config.get('stages', [])
        child_stages = child_config.get('stages', [])

        diff_analysis = {
            'reward': reward,
            'improvement_score': reward - (
                parent_config.get('metrics', {}).get('accuracy', 0) 
                if parent_config.get('metrics') else 0
            ),
            'major_changes': [],
            'detailed_differences': {},
            'change_summary': 'minor_changes'
        }
        
        # 1. 检查stage数量差异
        has_stage_count_change = False
        if len(parent_stages) != len(child_stages):
            diff_analysis['major_changes'].append(
                f"stage_count_changed: {len(parent_stages)} -> {len(child_stages)}"
            )
            has_stage_count_change = True
        
        # 2. 检查每个stage的block数量差异
        block_count_changes = []
        for i, (p_stage, c_stage) in enumerate(zip(parent_stages, child_stages)):
            p_blocks = p_stage.get('blocks', [])
            c_blocks = c_stage.get('blocks', [])
            
            if len(p_blocks) != len(c_blocks):
                block_count_changes.append(
                    f"stage_{i}_blocks: {len(p_blocks)} -> {len(c_blocks)}"
                )
        
        has_block_count_change = len(block_count_changes) > 0
        if block_count_changes:
            diff_analysis['major_changes'].extend(block_count_changes)
        
        # 3. 检查conv type差异
        conv_type_changes = []
        for i, (p_stage, c_stage) in enumerate(zip(parent_stages, child_stages)):
            p_blocks = p_stage.get('blocks', [])
            c_blocks = c_stage.get('blocks', [])
            
            for j, (p_block, c_block) in enumerate(zip(p_blocks, c_blocks)):
                p_conv_type = p_block.get('type', '')
                c_conv_type = c_block.get('type', '')
                
                if p_conv_type != c_conv_type:
                    conv_type_changes.append(
                        f"stage_{i}_block_{j}_conv_type: {p_conv_type} -> {c_conv_type}"
                    )
        
        has_conv_type_change = len(conv_type_changes) > 0
        if conv_type_changes:
            diff_analysis['major_changes'].extend(conv_type_changes)
        
        # 根据变化类型设置summary
        if diff_analysis['major_changes']:
            change_types = []
            if has_stage_count_change:
                change_types.append('structural')
            if has_block_count_change:
                change_types.append('block_count')
            if has_conv_type_change:
                change_types.append('conv_type')
            
            if change_types:
                if len(change_types) == 1:
                    diff_analysis['change_summary'] = f'major_{change_types[0]}_change'
                else:
                    diff_analysis['change_summary'] = (
                        f'major_combined_change_{"_".join(change_types)}'
                    )
            else:
                diff_analysis['change_summary'] = 'major_changes'
            
            return diff_analysis
        
        # 4. 详细分析其他参数差异
        detailed_diffs = {}
        has_significant_changes = False
        
        for i, (p_stage, c_stage) in enumerate(zip(parent_stages, child_stages)):
            stage_diffs = {}
            
            # 检查channels变化
            p_channels = p_stage.get('channels', 0)
            c_channels = c_stage.get('channels', 0)
            if p_channels != c_channels:
                stage_diffs['channels'] = f"{p_channels} -> {c_channels}"
                if abs(p_channels - c_channels) > 6:
                    has_significant_changes = True
            
            # 检查block内部参数
            p_blocks = p_stage.get('blocks', [])
            c_blocks = c_stage.get('blocks', [])
            block_diffs = []
            
            for j, (p_block, c_block) in enumerate(zip(p_blocks, c_blocks)):
                block_diff = {}
                
                # 检查kernel_size
                p_kernel = p_block.get('kernel_size', 3)
                c_kernel = c_block.get('kernel_size', 3)
                if p_kernel != c_kernel:
                    block_diff['kernel_size'] = f"{p_kernel} -> {c_kernel}"
                    if abs(p_kernel - c_kernel) > 1:
                        has_significant_changes = True
                
                # 检查stride
                p_stride = p_block.get('stride', 1)
                c_stride = c_block.get('stride', 1)
                if p_stride != c_stride:
                    block_diff['stride'] = f"{p_stride} -> {c_stride}"
                    has_significant_changes = True
                
                # 检查expansion
                p_expansion = p_block.get('expansion', 1)
                c_expansion = c_block.get('expansion', 1)
                if p_expansion != c_expansion:
                    block_diff['expansion'] = f"{p_expansion} -> {c_expansion}"
                    if abs(p_expansion - c_expansion) > 1:
                        has_significant_changes = True
                
                # 检查SE模块
                p_has_se = p_block.get('has_se', False)
                c_has_se = c_block.get('has_se', False)
                if p_has_se != c_has_se:
                    block_diff['has_se'] = f"{p_has_se} -> {c_has_se}"
                    has_significant_changes = True
                
                # 检查activation
                p_activation = p_block.get('activation', '')
                c_activation = c_block.get('activation', '')
                if p_activation != c_activation:
                    block_diff['activation'] = f"{p_activation} -> {c_activation}"
                
                if block_diff:
                    block_diffs.append({f'block_{j}': block_diff})
            
            if block_diffs:
                stage_diffs['blocks'] = block_diffs

            if stage_diffs:
                detailed_diffs[f'stage_{i}'] = stage_diffs
        
        diff_analysis['detailed_differences'] = detailed_diffs

        # 根据是否有显著变化更新summary
        if has_significant_changes:
            diff_analysis['change_summary'] = 'significant_parameter_changes'
        elif detailed_diffs:
            diff_analysis['change_summary'] = 'minor_parameter_changes'
        else:
            diff_analysis['change_summary'] = 'no_changes'
        
        return diff_analysis

    def get_best_candidate(self) -> Optional[CandidateModel]:
        """获取图中最佳候选模型"""
        if not self.nodes:
            return None
        
        evaluated_nodes = [node for node in self.nodes.values() 
                         if node.candidate and node.visit_count > 0]
        
        if not evaluated_nodes:
            return None
        
        best_node = max(evaluated_nodes, key=lambda node: node.average_reward)
        return best_node.candidate
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图结构统计信息"""
        total_nodes = len(self.nodes)
        
        # 统计所有边
        total_edges = 0
        for node in self.nodes.values():
            for direction, children_list in node.children.items():
                total_edges += len(children_list)
        
        forest_count = len(self.forest_roots)
        
        evaluated_nodes = [node for node in self.nodes.values() 
                         if node.visit_count > 0]
        avg_reward = (np.mean([node.average_reward for node in evaluated_nodes]) 
                     if evaluated_nodes else 0)
        best_reward = (max([node.average_reward for node in evaluated_nodes]) 
                      if evaluated_nodes else 0)
        
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "forest_count": forest_count,
            "evaluated_nodes": len(evaluated_nodes),
            "average_reward": avg_reward,
            "best_reward": best_reward,
            "global_insights_count": len(self.global_insights)
        }
    
    def save_graph_info(self, filepath: str):
        """
        保存图结构信息到文件
        
        参数:
            filepath: 保存路径
        """
        # 1. 计算每个节点的树深度和所属种子
        node_depths = {}
        node_root_seeds = {}

        def compute_depth_and_seed(node_id: str, current_depth: int = 0, 
                                   root_seed: str = None):
            """递归计算深度和所属种子"""
            if node_id in node_depths:
                return
            
            node = self.nodes.get(node_id)
            if not node:
                return
            
            if node.is_forest_root:
                root_seed = node_id
                current_depth = 0
            
            node_depths[node_id] = current_depth
            node_root_seeds[node_id] = root_seed
            
            # 递归处理子节点
            for direction, children_list in node.children.items():
                for child in children_list:
                    compute_depth_and_seed(child.node_id, current_depth + 1, root_seed)
        
        # 为所有森林根节点计算深度
        for root in self.forest_roots:
            compute_depth_and_seed(root.node_id)
        
        # 2. 构建节点信息
        nodes_info = {}
        for node_id, node in self.nodes.items():
            # 找到父节点信息
            parent_id = None
            parent_direction = None
            for potential_parent_id, potential_parent in self.nodes.items():
                for direction, children_list in potential_parent.children.items():
                    for child in children_list:
                        if child.node_id == node_id:
                            parent_id = potential_parent_id
                            parent_direction = direction
                            break
                    if parent_id:
                        break
                if parent_id:
                    break
            
            # 提取内存信息
            memory_mb = 0.0
            if node.candidate and node.candidate.metrics:
                if 'quantized_memory' in node.candidate.metrics:
                    memory_mb = node.candidate.metrics.get('quantized_memory', 0.0)
                elif 'original_memory' in node.candidate.metrics:
                    memory_mb = node.candidate.metrics.get('original_memory', 0.0)
            
            # 构建节点信息
            nodes_info[node_id] = {
                "node_id": node_id,
                "candidate_config": node.candidate.config if node.candidate else None,
                "task_type": node.candidate.task_type if node.candidate else None,  # ✅ 添加task_type
                "metrics": {
                    "accuracy": (node.candidate.metrics.get('accuracy', 0.0) 
                               if node.candidate else 0.0),
                    "memory_mb": memory_mb,
                    "quantization_mode": (
                        node.candidate.config.get('quant_mode', 'none') 
                        if node.candidate and node.candidate.config else 'none'
                    )
                },
                "visit_count": node.visit_count,
                "average_reward": node.average_reward,
                "children_directions": list(node.children.keys()),
                "direction_stats": {
                    direction: {
                        "q_value": node.direction_q_values.get(direction, 0.0),
                        "visits": node.direction_visits.get(direction, 0)
                    }
                    for direction in node.directions
                },
                "tree_metadata": {
                    "depth": node_depths.get(node_id, 0),
                    "root_seed_id": node_root_seeds.get(node_id, None),
                    "iteration": getattr(node, 'iteration', None),
                    "is_forest_root": node.is_forest_root
                },
                "graph_info": {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "parent_direction": parent_direction,
                    "children_count": len(node.children),
                    "children_ids": [
                        child.node_id 
                        for children_list in node.children.values() 
                        for child in children_list
                    ],
                    "is_forest_root": node.is_forest_root
                }
            }

        # 3. 构建完整的图结构数据
        import json
        graph_data = {
            "statistics": self.get_graph_statistics(),
            "forest_roots": [root.node_id for root in self.forest_roots],
            "nodes": nodes_info,
            "global_insights": self.global_insights,
            "training_trajectories": self._extract_training_trajectories(nodes_info)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _extract_training_trajectories(self, nodes_info: dict) -> list:
        """
        提取用于LLaMA3训练的轨迹数据
        
        参数:
            nodes_info: 节点信息字典
        
        返回:
            轨迹列表
        """
        trajectories = []
        
        for node_id, node_data in nodes_info.items():
            # 跳过森林根节点
            if node_data['graph_info']['parent_id'] is None:
                continue
            
            parent_id = node_data['graph_info']['parent_id']
            parent_direction = node_data['graph_info']['parent_direction']
            
            if parent_id in nodes_info:
                parent_data = nodes_info[parent_id]
                
                # 构建训练样本
                trajectory = {
                    "trajectory_id": f"{parent_id}_to_{node_id}",
                    "parent": {
                        "node_id": parent_id,
                        "config": parent_data['candidate_config'],
                        "task_type": parent_data.get('task_type'),  # ✅ 添加task_type
                        "reward": parent_data['average_reward'],
                        "visit_count": parent_data['visit_count']
                    },
                    "action": {
                        "direction": parent_direction,
                        "q_value": parent_data['direction_stats'].get(
                            parent_direction, {}
                        ).get('q_value', 0.0)
                    },
                    "child": {
                        "node_id": node_id,
                        "config": node_data['candidate_config'],
                        "task_type": node_data.get('task_type'),  # ✅ 添加task_type
                        "reward": node_data['average_reward'],
                        "metrics": node_data['metrics']
                    },
                    "context": {
                        "depth": node_data['tree_metadata']['depth'],
                        "root_seed_id": node_data['tree_metadata']['root_seed_id'],
                        "iteration": node_data['tree_metadata']['iteration']
                    }
                }
                
                trajectories.append(trajectory)
        
        return trajectories