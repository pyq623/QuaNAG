# /root/unified_nas/mcts/__init__.py
"""
统一的MCTS搜索模块
支持HAR和Image任务的神经架构搜索
"""

from .mcts_nodes import MCTSNode, MCTSTree
from .parallel_evaluator import ParallelModelEvaluator, worker_process
from .degradation_strategies import ConfigDegradationManager
from .sfs_parallel import create_parallel_sfs_search
__all__ = [
    'MCTSNode',
    'MCTSTree',
    'ParallelModelEvaluator',
    'worker_process',
    'ConfigDegradationManager',
    'create_parallel_sfs_search'
]

__version__ = '1.0.0'
__author__ = 'Pupper Yard'
__description__ = 'Unified MCTS for HAR and Image NAS'