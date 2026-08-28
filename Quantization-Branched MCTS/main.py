#!/usr/bin/env python3
"""
统一的并行散射森林搜索主程序
支持 HAR (1D) 和 Image (2D) 任务

使用示例:
    # HAR任务
    python main_parallel.py --dataset MMAct --iterations 100 --num_gpus 4
    
    # Image任务
    python main_parallel.py --dataset cifar10 --iterations 100 --num_gpus 4
"""

import os
import json
import argparse
import time
from datetime import datetime, timedelta
from configs import get_search_space
import glob

# 设置时区为中国北京时区
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

# 导入并行搜索类
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mcts import create_parallel_sfs_search
from data import get_dataset_info


class Timer:
    """计时器类"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
        
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """退出 with 语句时自动停止"""
        self.stop()
    
    def get_elapsed_str(self) -> str:
        """获取格式化的耗时字符串"""
        return self.format_time(self.elapsed)
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}分钟 ({seconds:.1f}秒)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f}小时 ({minutes:.1f}分钟)"


def setup_search_space(dataset_name: str = "MMAct", max_memory: float = None) -> tuple:
    """
    设置搜索空间
    
    Args:
        dataset_name: 数据集名称
        max_memory: 最大峰值内存限制 (MB)，如果为None则使用配置文件中的默认值
    
    Returns:
        (search_space, constraints)
    """
    info = get_search_space()
    search_space = info['search_space']
    constraints = info['constraints']

    # 如果指定了 max_memory，覆盖默认值
    if max_memory is not None:
        # 将 MB 转换为字节
        constraints['max_peak_memory'] = max_memory
        print(f"✅ 使用自定义内存限制: {max_memory} MB ")
    else:
        # 显示默认值
        default_value = constraints['max_peak_memory']
        # 如果是字符串，转换为浮点数
        if isinstance(default_value, str):
            default_value = float(default_value)
            constraints['max_peak_memory'] = default_value
        
        default_mb = default_value
        print(f"ℹ️  使用默认内存限制: {default_mb} MB ")

    return search_space, constraints


def save_search_results(search, output_dir: str, args: dict, 
                       timing_stats: dict, search_status: dict = None):
    """
    保存搜索结果
    
    Args:
        search: 搜索实例
        output_dir: 输出目录
        args: 实验参数
        timing_stats: 计时统计
        search_status: 搜索状态
    """
    # 添加搜索状态标记
    if search_status:
        timing_stats['search_status'] = search_status

    # 保存最佳模型配置
    best_model = search.get_best_model()
    if best_model:
        # 从树中查找对应节点以获取深度信息
        best_node = None
        for node_id, node in search.tree.nodes.items():
            if node.candidate and \
               node.candidate.candidate_id == best_model.candidate_id:
                best_node = node
                break
        
        # 计算深度
        depth = 0
        if best_node:
            current = best_node
            while current.parent and not current.parent.is_forest_root:
                depth += 1
                current = current.parent
            if current.parent:
                depth += 1
        
        peak_memory = best_model.metrics.get(
            'quantized_memory',
            best_model.metrics.get('original_memory', 0.0)
        )
        
        # 收集完整的元信息
        best_model_data = {
            "config": best_model.config,
            "performance": {
                "reward": search.best_reward,
                "accuracy": best_model.metrics.get('accuracy', 0.0),
                "peak_memory": peak_memory,
                "gpu_id": best_model.metrics.get('gpu_id', -1)
            },
            "metadata": {
                "candidate_id": best_model.candidate_id \
                    if hasattr(best_model, 'candidate_id') else 'unknown',
                "iteration_discovered": getattr(best_model, 'iteration', 'unknown'),
                "parent_seed_id": getattr(best_model, 'root_seed_id', 'unknown'),
                "depth_from_seed": depth,
                "parent_node_id": getattr(best_model, 'parent_id', None),
                "parent_direction": getattr(best_model, 'parent_direction', None),
                "task_type": best_model.task_type \
                    if hasattr(best_model, 'task_type') else 'unknown'
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        best_config_path = os.path.join(output_dir, "best_model_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_model_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 最佳模型配置已保存: {best_config_path}")
    else:
        print(f"⚠️  未找到最佳模型（可能搜索被提前终止）")

    # 保存搜索统计信息
    stats = search.get_search_statistics()
    stats['timing'] = timing_stats
    
    stats_path = os.path.join(output_dir, "search_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ 搜索统计已保存: {stats_path}")
    
    # 保存搜索状态
    state_path = os.path.join(output_dir, "search_state.json")
    try:
        search.save_search_state(state_path)
        print(f"✅ 搜索状态已保存: {state_path}")
    except Exception as e:
        print(f"⚠️  保存搜索状态失败: {e}")
    
    # 保存实验参数
    experiment_data = {
        'args': args,
        'timing': timing_stats,
        'search_status': search_status
    }
    args_path = os.path.join(output_dir, "experiment_args.json")
    with open(args_path, 'w') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 实验参数已保存: {args_path}")

    # 保存对话记录
    print("\n💾 保存对话记录...")
    try:
        search.save_dialogue_records(output_dir)
    except Exception as e:
        print(f"⚠️ 保存对话记录失败: {e}")
    
    # 保存计时报告
    timing_report_path = os.path.join(output_dir, "timing_report.txt")
    with open(timing_report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("⏱️  运行时间统计报告\n")
        f.write("="*60 + "\n\n")

        # 搜索状态信息
        if search_status:
            f.write("📊 搜索状态:\n")
            f.write(f"  - 完成迭代: {search_status.get('completed_iterations', 0)}/"
                   f"{search_status.get('total_iterations', 0)}\n")
            if search_status.get('interrupted'):
                f.write(f"  - 状态: ⚠️  用户中断\n")
            elif search_status.get('timeout_triggered'):
                f.write(f"  - 状态: ⚠️  超时退出\n")
            else:
                f.write(f"  - 状态: ✅ 正常完成\n")
            f.write(f"  - 待处理任务: {search_status.get('pending_tasks', 0)}\n\n")
        
        f.write(f"🕐 总运行时间: {Timer.format_time(timing_stats['total_time'])}\n\n")
        
        f.write("📊 各阶段耗时:\n")
        if 'initialization_time' in timing_stats:
            f.write(f"  1. 初始化阶段: "
                   f"{Timer.format_time(timing_stats['initialization_time'])}\n")
        if 'forest_init_time' in timing_stats:
            f.write(f"  2. 森林初始化: "
                   f"{Timer.format_time(timing_stats['forest_init_time'])}\n")
        if 'search_time' in timing_stats:
            f.write(f"  3. 搜索执行:   "
                   f"{Timer.format_time(timing_stats['search_time'])}\n")
        if 'save_time' in timing_stats:
            f.write(f"  4. 结果保存:   "
                   f"{Timer.format_time(timing_stats['save_time'])}\n\n")
        
        f.write("📈 搜索效率:\n")
        actual_iterations = search_status.get('completed_iterations', 
                          timing_stats.get('total_iterations', 0)) \
                          if search_status else timing_stats.get('total_iterations', 0)
        f.write(f"  - 计划迭代次数: {timing_stats.get('total_iterations', 0)}\n")
        f.write(f"  - 实际完成次数: {actual_iterations}\n")

        if actual_iterations > 0 and 'search_time' in timing_stats and \
           timing_stats['search_time'] > 0:
            avg_iter_time = timing_stats['search_time'] / actual_iterations
            f.write(f"  - 平均每次迭代: {Timer.format_time(avg_iter_time)}\n")
            throughput = 3600 / avg_iter_time
            f.write(f"  - 吞吐量: {throughput:.2f} 次迭代/小时\n")
        
        f.write(f"  - GPU数量: {timing_stats['num_gpus']}\n")
        f.write(f"  - GPU利用率估算: "
               f"{timing_stats.get('gpu_utilization_estimate', 'N/A')}\n\n")
        
        f.write("⏰ 时间戳:\n")
        f.write(f"  - 开始时间: {timing_stats['start_time']}\n")
        f.write(f"  - 结束时间: {timing_stats['end_time']}\n")
        
    print(f"✅ 计时报告已保存: {timing_report_path}")


def print_timing_summary(timing_stats: dict, search_status: dict = None):
    """打印计时摘要"""
    print("\n" + "="*60)
    print("⏱️  运行时间统计")
    print("="*60)
    
    # 显示搜索状态
    if search_status:
        print(f"\n📊 搜索状态:")
        print(f"  完成迭代: {search_status.get('completed_iterations', 0)}/"
              f"{search_status.get('total_iterations', 0)}")
        if search_status.get('interrupted'):
            print(f"  状态: ⚠️  用户中断")
        elif search_status.get('timeout_triggered'):
            print(f"  状态: ⚠️  超时退出")
        else:
            print(f"  状态: ✅ 正常完成")
        print(f"  待处理任务: {search_status.get('pending_tasks', 0)}")
    
    print(f"\n🕐 总运行时间: {Timer.format_time(timing_stats['total_time'])}")
    
    print(f"\n📊 各阶段耗时:")
    if 'initialization_time' in timing_stats:
        print(f"  1️⃣  初始化阶段: "
              f"{Timer.format_time(timing_stats['initialization_time'])} "
              f"({timing_stats['initialization_time']/timing_stats['total_time']*100:.1f}%)")
    
    if 'forest_init_time' in timing_stats:
        print(f"  2️⃣  森林初始化: "
              f"{Timer.format_time(timing_stats['forest_init_time'])} "
              f"({timing_stats['forest_init_time']/timing_stats['total_time']*100:.1f}%)")
    
    if 'search_time' in timing_stats:
        print(f"  3️⃣  搜索执行:   "
              f"{Timer.format_time(timing_stats['search_time'])} "
              f"({timing_stats['search_time']/timing_stats['total_time']*100:.1f}%)")
    
    if 'save_time' in timing_stats:
        print(f"  4️⃣  结果保存:   "
              f"{Timer.format_time(timing_stats['save_time'])} "
              f"({timing_stats['save_time']/timing_stats['total_time']*100:.1f}%)")
    
    print(f"\n📈 搜索效率:")
    actual_iterations = search_status.get('completed_iterations', 
                      timing_stats.get('total_iterations', 0)) \
                      if search_status else timing_stats.get('total_iterations', 0)
    print(f"  - 计划迭代次数: {timing_stats.get('total_iterations', 0)}")
    print(f"  - 实际完成次数: {actual_iterations}")
    
    if actual_iterations > 0 and 'search_time' in timing_stats and \
       timing_stats['search_time'] > 0:
        avg_iter_time = timing_stats['search_time'] / actual_iterations
        print(f"  - 平均每次迭代: {Timer.format_time(avg_iter_time)}")
        throughput = 3600 / avg_iter_time
        print(f"  - 吞吐量: {throughput:.2f} 次迭代/小时")

    print(f"  - GPU数量: {timing_stats['num_gpus']}")

    if timing_stats['num_gpus'] > 0 and 'search_time' in timing_stats and \
       actual_iterations > 0:
        ideal_time = timing_stats['search_time'] / timing_stats['num_gpus']
        actual_avg_time = timing_stats['search_time'] / actual_iterations
        utilization = (ideal_time / actual_avg_time) * 100 \
                     if actual_avg_time > 0 else 0
        print(f"  - GPU利用率估算: {utilization:.1f}%")
        timing_stats['gpu_utilization_estimate'] = f"{utilization:.1f}%"
    
    print(f"\n⏰ 时间范围:")
    print(f"  开始: {timing_stats['start_time']}")
    print(f"  结束: {timing_stats['end_time']}")
    
    print("="*60)


def print_final_results(search, timing_stats: dict, search_status: dict = None):
    """打印最终结果"""
    print("\n" + "="*60)
    # 根据搜索状态显示不同的标题
    if search_status:
        if search_status.get('interrupted'):
            print("⚠️  搜索中断! 当前结果:")
        elif search_status.get('timeout_triggered'):
            print("⚠️  搜索超时! 当前结果:")
        else:
            print("🎉 搜索完成! 最终结果:")
    else:
        print("🎉 搜索完成! 最终结果:")

    print("="*60)
    
    best_model = search.get_best_model()
    stats = search.get_search_statistics()
    
    if best_model:
        print(f"\n🏆 最佳模型:")
        print(f"   - 奖励: {stats['best_reward']:.4f}")
        print(f"   - 配置ID: {best_model.candidate_id if hasattr(best_model, 'candidate_id') else 'N/A'}")
        print(f"   - 任务类型: {stats.get('task_type', 'unknown').upper()}")
        
        print(f"\n📊 搜索统计:")
        actual_iters = search_status.get('completed_iterations', stats['iterations']) \
                      if search_status else stats['iterations']
        print(f"   - 实际迭代次数: {actual_iters}")
        print(f"   - 唯一配置: {stats['unique_configs']}")
        print(f"   - 重复次数: {stats['duplicate_count']}")

        total_attempts = actual_iters + stats['duplicate_count']
        if total_attempts > 0:
            print(f"   - 重复率: {stats['duplicate_count']/total_attempts*100:.1f}%")

        print(f"\n🌳 搜索树统计:")
        print(f"   - 总节点数: {stats['tree_statistics']['total_nodes']}")
        print(f"   - 评估节点: {stats['tree_statistics']['evaluated_nodes']}")
        print(f"   - 森林大小: {stats['tree_statistics']['forest_count']}")
        print(f"   - 平均奖励: {stats['tree_statistics']['average_reward']:.4f}")
        print(f"   - 最佳奖励: {stats['tree_statistics']['best_reward']:.4f}")
        
        print(f"\n🔍 全局经验洞察:")
        insights = stats['global_insights']
        if insights:
            for direction, insight in insights.items():
                if direction.startswith('direction_'):
                    direction_name = direction.replace('direction_', '')
                    success_rate = insight.get('success_rate', 0)
                    avg_reward = insight.get('average_reward', 0)
                    visit_count = insight.get('visit_count', 0)
                    print(f"   - {direction_name:6s}: 访问={visit_count:3d}, "
                          f"成功率={success_rate:.1%}, 平均奖励={avg_reward:.3f}")
        else:
            print("   (暂无洞察数据)")
    else:
        print(f"\n⚠️  未找到最佳模型")
        print(f"   这可能是因为搜索被提前终止")
    
    # 打印计时摘要
    print_timing_summary(timing_stats, search_status)
    
    print("\n" + "="*60)


def get_available_datasets():
    """
    获取可用的数据集列表
    
    Returns:
        dict: {task_type: [dataset_names]}
    """
    # 这里硬编码一些常用数据集
    # 实际使用时可以动态扫描数据目录
    return {
        'har': ['MMAct', 'Mhealth', 'Wharf', 'UTD-MHAD', 'USCHAD'],
        'image': ['cifar10', 'cifar100', 'mnist']
    }


def main():
    """主函数"""
    # 获取可用数据集
    available_datasets = get_available_datasets()
    all_datasets = []
    for datasets in available_datasets.values():
        all_datasets.extend(datasets)
    
    parser = argparse.ArgumentParser(
        description="统一的并行散射森林搜索（支持 HAR 和 Image 任务）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        HAR任务:
            python main_parallel.py --dataset MMAct --iterations 100 --num_gpus 4
        
        Image任务:
            python main_parallel.py --dataset cifar10 --iterations 100 --num_gpus 40  --timeout 7500
        
        断点恢复:
            python main_parallel.py --dataset cifar10 --resume  <xxx.pkl>
                """
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=all_datasets,
                       help="数据集名称（会自动推断任务类型）")
    parser.add_argument("--iterations", type=int, default=50,
                       help="搜索迭代次数")
    parser.add_argument("--num_seeds", type=int, default=4,
                       help="初始种子数量")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="使用的GPU数量")
    parser.add_argument("--train_epochs", type=int, default=200,
                       help="每个模型的训练轮数")
    parser.add_argument("--exploration_weight", type=float, default=6.0,
                       help="探索权重")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="实验名称（不指定则自动生成）")
    parser.add_argument("--timeout", type=int, default=600,
                       help="超时时间（秒），无新结果达到此时间后自动保存退出")
    parser.add_argument("--max_memory", type=float, default=None,
                       help="最大峰值内存限制 (MB)，不指定则使用配置文件默认值")
    parser.add_argument("--resume", type=str, default=None,
                       help="从指定断点文件恢复")
    parser.add_argument("--auto_resume", action="store_true",
                       help="自动从最新的断点恢复")
    parser.add_argument("--max_runtime", type=float, default=None,
                   help="最大运行时长（小时），例如 4.0 表示4小时")
    
    args = parser.parse_args()
    
    # ✅ 自动推断任务类型
    try:
        dataset_info = get_dataset_info(args.dataset)
        task_type = dataset_info.get('task_type', 'unknown')
    except:
        task_type = 'unknown'
    
    if task_type == 'har':
        args.timeout = 300
        # args.num_gpus = 4 # HAR任务通常使用较少GPU
    elif task_type == 'image':
        args.timeout = 7500
        # args.num_gpus = 40 # Image任务通常使用更多GPU

    # ✅ 如果没有指定实验名称，自动生成
    if args.experiment_name is None:
        args.experiment_name = f"sfs_{task_type}_{args.dataset}"
    
    # 统一的时间戳
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 记录开始时间
    experiment_start_time = time.time()
    start_datetime = datetime.now()
    
    print("🚀 开始统一并行散射森林搜索（支持 HAR 和 Image 任务）...")
    print(f"⏰ 开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📋 实验配置:")
    print(f"   - 数据集: {args.dataset}")
    print(f"   - 任务类型: {task_type.upper()}")
    print(f"   - 迭代次数: {args.iterations}")
    print(f"   - 初始种子: {args.num_seeds}")
    print(f"   - GPU数量: {args.num_gpus}")
    print(f"   - 训练轮数: {args.train_epochs}")
    print(f"   - 探索权重: {args.exploration_weight}")
    print(f"   - 实验名称: {args.experiment_name}")
    print(f"   - 超时设置: {args.timeout}秒")
    print(f"   - 最大内存限制: "
          f"{args.max_memory if args.max_memory else '使用默认值'} MB")
    print(f"   - 断点恢复: "
          f"{'自动最新' if args.auto_resume else (args.resume if args.resume else '不启用')}")
    print("\n" + "="*60)
    
    # 初始化计时统计
    timing_stats = {
        'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'num_gpus': args.num_gpus,
        'total_iterations': args.iterations,
        'experiment_timestamp': experiment_timestamp,
        'task_type': task_type
    }

    # 初始化搜索状态
    search_status = None
    search = None
    output_dir = None
    
    # 设置搜索空间和约束
    print("\n⏱️  [1/4] 初始化阶段...")
    with Timer("initialization") as init_timer:
        search_space, constraints = setup_search_space(args.dataset, args.max_memory)
        
        # 使用统一时间戳创建目录
        output_dir = f"results/{args.experiment_name}_{experiment_timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_dir = f"sfs_checkpoints/{experiment_timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"📁 输出目录: {output_dir}")
        print(f"💾 Checkpoint目录: {checkpoint_dir}")
    
    timing_stats['initialization_time'] = init_timer.elapsed
    print(f"   ✅ 初始化完成，耗时: {init_timer.get_elapsed_str()}")
    
    try:
        # 创建并行搜索实例
        search = create_parallel_sfs_search(
            search_space=search_space,
            constraints=constraints,
            dataset_name=args.dataset,
            num_gpus=args.num_gpus,
            exploration_weight=args.exploration_weight,
            train_epochs=args.train_epochs,
            timeout_seconds=args.timeout,
            checkpoint_dir=checkpoint_dir
        )
        
        # ===== 断点恢复逻辑 =====
        skip_forest_init = False

        if args.auto_resume or args.resume:
            if args.auto_resume:
                # 自动查找最新断点
                checkpoint_pattern = os.path.join(
                    search.checkpoint_dir, "checkpoint_*.pkl"
                )
                checkpoint_files = glob.glob(checkpoint_pattern)
                
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                    print(f"\n💾 自动恢复模式")
                    print(f"   找到 {len(checkpoint_files)} 个断点文件")
                    print(f"   加载最新断点: {os.path.basename(latest_checkpoint)}")
                    
                    if search.load_checkpoint(latest_checkpoint):
                        print(f"✅ 断点恢复成功，跳过森林初始化")
                        skip_forest_init = True
                    else:
                        print(f"⚠️  断点加载失败，从头开始")
                else:
                    print(f"\n⚠️  未找到断点文件，从头开始")
                    print(f"   搜索目录: {search.checkpoint_dir}")
            
            elif args.resume:
                # 手动指定断点
                print(f"\n💾 手动恢复模式")
                print(f"   加载断点: {args.resume}")
                
                if os.path.exists(args.resume):
                    if search.load_checkpoint(args.resume):
                        print(f"✅ 断点恢复成功，跳过森林初始化")
                        skip_forest_init = True
                    else:
                        print(f"❌ 断点加载失败，退出程序")
                        return
                else:
                    print(f"❌ 断点文件不存在: {args.resume}")
                    return

        # 初始化森林
        if not skip_forest_init:
            print(f"\n⏱️  [2/4] 森林初始化阶段 ({args.num_seeds} 个种子)...")
            with Timer("forest_init") as forest_timer:
                search.initialize_forest(num_seeds=args.num_seeds)
            
            timing_stats['forest_init_time'] = forest_timer.elapsed
            timing_stats['avg_seed_time'] = forest_timer.elapsed / args.num_seeds
            print(f"   ✅ 森林初始化完成，耗时: {forest_timer.get_elapsed_str()}")
            print(f"   📊 平均每个种子: "
                  f"{Timer.format_time(timing_stats['avg_seed_time'])}")
        else:
            # 断点恢复时启动工作进程
            print(f"\n⏭️  [2/4] 跳过森林初始化（已从断点恢复）")
            print(f"   启动工作进程...")
            search._start_workers()
            timing_stats['forest_init_time'] = 0
            timing_stats['avg_seed_time'] = 0

        # 执行搜索
        print(f"\n⏱️  [3/4] 搜索执行阶段 ({args.iterations} 次迭代)...")
        if timing_stats.get('avg_seed_time', 0) > 0:
            print(f"   预计时间: "
                  f"{Timer.format_time(timing_stats['avg_seed_time'] * args.iterations)}")
        
        with Timer("search") as search_timer:
            search_status = search.search(
                iterations=args.iterations,
                exploration_weight=args.exploration_weight,
                max_runtime_hours=args.max_runtime  # ✅ 传入运行时长
            )
        
        timing_stats['search_time'] = search_timer.elapsed
        actual_iterations = search_status.get('completed_iterations', args.iterations) \
                           if search_status else args.iterations
        
        if actual_iterations > 0:
            timing_stats['avg_iteration_time'] = \
                search_timer.elapsed / actual_iterations
            print(f"   ✅ 搜索完成，耗时: {search_timer.get_elapsed_str()}")
            print(f"   📊 平均每次迭代: "
                  f"{Timer.format_time(timing_stats['avg_iteration_time'])}")
            print(f"   📊 实际完成: {actual_iterations}/{args.iterations} 次迭代")
        else:
            timing_stats['avg_iteration_time'] = 0
            print(f"   ⚠️  搜索提前终止，未完成任何迭代")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断实验 (Ctrl+C)")
        if search_status is None:
            search_status = {
                'completed_iterations': 0,
                'total_iterations': args.iterations,
                'interrupted': True,
                'timeout_triggered': False,
                'pending_tasks': 0
            }
        else:
            search_status['interrupted'] = True
        
        timing_stats['interrupted'] = True

        if search:
            try:
                stats = search.get_search_statistics()
                search_status['completed_iterations'] = stats.get('iterations', 0)
            except:
                pass

    except Exception as e:
        print(f"\n❌ 搜索过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        if search_status is None:
            search_status = {
                'completed_iterations': 0,
                'total_iterations': args.iterations,
                'interrupted': False,
                'timeout_triggered': False,
                'pending_tasks': 0,
                'error': str(e)
            }
        
        timing_stats['error'] = str(e)

    finally:
        # 确保在任何情况下都保存结果
        print("\n🧹 清理资源并保存结果...")
        
        # 计算总时间
        timing_stats['total_time'] = time.time() - experiment_start_time
        timing_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
        # 保存结果
        if search and output_dir:
            print(f"\n⏱️  [4/4] 保存结果...")
            with Timer("save") as save_timer:
                timing_stats['save_time'] = 0
                try:
                    save_search_results(search, output_dir, vars(args),
                                       timing_stats, search_status)
                except Exception as save_error:
                    print(f"❌ 保存结果时出错: {save_error}")
                    import traceback
                    traceback.print_exc()
            
            timing_stats['save_time'] = save_timer.elapsed
            print(f"   ✅ 结果保存完成，耗时: {save_timer.get_elapsed_str()}")
                    
            # 打印最终结果
            try:
                print_final_results(search, timing_stats, search_status)
            except Exception as print_error:
                print(f"⚠️  打印结果时出错: {print_error}")
            
            # 打印总结
            end_datetime = datetime.now()
            is_complete = not (search_status and (
                search_status.get('interrupted') or
                search_status.get('timeout_triggered')
            ))
            
            print(f"\n{'✅' if is_complete else '⚠️ '} "
                  f"实验{'完成' if is_complete else '终止'}!")
            print(f"📁 结果保存在: {output_dir}")
            print(f"⏰ 结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  总耗时: {Timer.format_time(timing_stats['total_time'])}")
            
            # 提示信息
            if search_status:
                if search_status.get('interrupted'):
                    print(f"\n💡 提示: 搜索被用户中断，当前结果已保存")
                    print(f"   可以通过减少 --iterations 参数继续后续实验")
                elif search_status.get('timeout_triggered'):
                    print(f"\n💡 提示: 搜索因超时自动终止，当前结果已保存")
                    print(f"   可以通过增加 --timeout 参数或检查GPU状态来解决")
            
            # 估算运行更多迭代需要的时间
            if search_status and not search_status.get('timeout_triggered'):
                completed = search_status.get('completed_iterations', 0)
                if completed > 0 and completed < 100 and \
                   'avg_iteration_time' in timing_stats:
                    estimated_100 = (timing_stats['avg_iteration_time'] * 100) + \
                                   timing_stats.get('forest_init_time', 0)
                    print(f"\n💡 提示: 如果运行100次迭代，预计需要: "
                          f"{Timer.format_time(estimated_100)}")
        else:
            print(f"\n⚠️  无法保存结果（搜索实例或输出目录未创建）")


if __name__ == "__main__":
    main()