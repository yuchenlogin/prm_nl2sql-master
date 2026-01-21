#!/usr/bin/env python3
"""
智能GPU调度器
自动选择显存使用率低的GPU进行训练
"""

import subprocess
import sys
import os
import argparse
from typing import List, Dict, Tuple
import re


class GPUScheduler:
    """智能GPU调度器"""

    def __init__(self, memory_threshold: float = 30.0, min_gpus: int = 1, max_gpus: int = 8):
        """
        初始化GPU调度器

        Args:
            memory_threshold: 显存使用率阈值（百分比），低于此值的GPU会被选中
            min_gpus: 最少需要的GPU数量
            max_gpus: 最多使用的GPU数量
        """
        self.memory_threshold = memory_threshold
        self.min_gpus = min_gpus
        self.max_gpus = max_gpus

    def get_gpu_info(self) -> List[Dict]:
        """
        获取所有GPU的信息

        Returns:
            GPU信息列表，每个元素包含 gpu_id, memory_used_mb, memory_total_mb, memory_percent, utilization
        """
        try:
            # 使用nvidia-smi获取GPU信息
            cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = [x.strip() for x in line.split(',')]
                if len(parts) == 4:
                    gpu_id = int(parts[0])
                    memory_used = int(parts[1])
                    memory_total = int(parts[2])
                    utilization = int(parts[3])
                    memory_percent = (memory_used / memory_total) * 100.0

                    gpu_info.append({
                        'gpu_id': gpu_id,
                        'memory_used_mb': memory_used,
                        'memory_total_mb': memory_total,
                        'memory_percent': memory_percent,
                        'utilization': utilization
                    })

            return gpu_info

        except subprocess.CalledProcessError as e:
            print(f"错误: 无法获取GPU信息 - {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"错误: 解析GPU信息失败 - {e}", file=sys.stderr)
            return []

    def select_gpus(self) -> List[int]:
        """
        选择合适的GPU

        Returns:
            选中的GPU ID列表
        """
        gpu_info = self.get_gpu_info()

        if not gpu_info:
            print("错误: 没有找到可用的GPU", file=sys.stderr)
            return []

        # 筛选出显存使用率低于阈值的GPU
        available_gpus = [gpu for gpu in gpu_info if gpu['memory_percent'] < self.memory_threshold]

        if len(available_gpus) < self.min_gpus:
            print(f"警告: 只有 {len(available_gpus)} 个GPU显存使用率低于 {self.memory_threshold}%，"
                  f"需要至少 {self.min_gpus} 个GPU", file=sys.stderr)

            # 如果可用的GPU不够，选择显存使用率最低的GPU
            sorted_gpus = sorted(gpu_info, key=lambda x: x['memory_percent'])
            selected_ids = [gpu['gpu_id'] for gpu in sorted_gpus[:self.min_gpus]]
            print(f"回退方案: 选择了显存使用率最低的 {len(selected_ids)} 个GPU", file=sys.stderr)
            return selected_ids

        # 按显存使用率排序
        available_gpus.sort(key=lambda x: x['memory_percent'])

        # 选择最多max_gpus个GPU
        selected_ids = [gpu['gpu_id'] for gpu in available_gpus[:self.max_gpus]]

        return selected_ids

    def set_cuda_visible_devices(self, gpu_ids: List[int]):
        """
        设置CUDA_VISIBLE_DEVICES环境变量

        Args:
            gpu_ids: GPU ID列表
        """
        if gpu_ids:
            gpu_str = ','.join(map(str, gpu_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            print(f"✅ 已设置 CUDA_VISIBLE_DEVICES={ gpu_str }")
        else:
            print("⚠️ 没有可用的GPU")

    def print_gpu_status(self):
        """打印GPU状态"""
        gpu_info = self.get_gpu_info()

        print("\n" + "=" * 80)
        print("GPU 状态监控")
        print("=" * 80)
        print(f"{'GPU ID':<10}{'显存使用':<15}{'显存总量':<15}{'显存占用率':<15}{'GPU利用率':<15}")
        print("-" * 80)

        for gpu in gpu_info:
            memory_percent_str = f"{gpu['memory_percent']:.1f}%"

            # 标记低于阈值的GPU
            if gpu['memory_percent'] < self.memory_threshold:
                status = "✓ 可用"
            else:
                status = "✗ 繁忙"

            print(f"{gpu['gpu_id']:<10}{gpu['memory_used_mb']/1024:<15.2f}"
                  f"{gpu['memory_total_mb']/1024:<15.2f}{memory_percent_str:<15}"
                  f"{gpu['utilization']}%{'':<10}{status}")

        print("=" * 80 + "\n")

    def get_config(self) -> Dict:
        """
        获取GPU配置信息

        Returns:
            包含GPU配置的字典
        """
        selected_gpus = self.select_gpus()
        gpu_info = self.get_gpu_info()

        selected_gpu_info = [gpu for gpu in gpu_info if gpu['gpu_id'] in selected_gpus]

        config = {
            'CUDA_VISIBLE_DEVICES': ','.join(map(str, selected_gpus)),
            'N_GPUS_PER_NODE': len(selected_gpus),
            'TENSOR_MODEL_PARALLEL_SIZE': min(len(selected_gpus), 2),  # 最多为2
            'selected_gpu_ids': selected_gpus,
            'selected_gpu_info': selected_gpu_info
        }

        return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能GPU调度器')
    parser.add_argument('--memory-threshold', type=float, default=30.0,
                        help='显存使用率阈值（百分比），默认30%%')
    parser.add_argument('--min-gpus', type=int, default=1,
                        help='最少需要的GPU数量，默认1')
    parser.add_argument('--max-gpus', type=int, default=8,
                        help='最多使用的GPU数量，默认8')
    parser.add_argument('--set-env', action='store_true',
                        help='直接设置CUDA_VISIBLE_DEVICES环境变量')
    parser.add_argument('--get-config', action='store_true',
                        help='输出配置信息（可用于脚本解析）')
    parser.add_argument('--status', action='store_true',
                        help='只显示GPU状态，不进行调度')

    args = parser.parse_args()

    scheduler = GPUScheduler(
        memory_threshold=args.memory_threshold,
        min_gpus=args.min_gpus,
        max_gpus=args.max_gpus
    )

    # 打印GPU状态
    scheduler.print_gpu_status()

    if args.status:
        sys.exit(0)

    # 选择GPU
    selected_gpus = scheduler.select_gpus()

    if not selected_gpus:
        print("错误: 没有可用的GPU", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ 选择了 {len(selected_gpus)} 个GPU: {selected_gpus}")

    if args.get_config:
        config = scheduler.get_config()
        print("\n配置信息（供脚本使用）:")
        print(f"CUDA_VISIBLE_DEVICES={config['CUDA_VISIBLE_DEVICES']}")
        print(f"N_GPUS_PER_NODE={config['N_GPUS_PER_NODE']}")
        print(f"TENSOR_MODEL_PARALLEL_SIZE={config['TENSOR_MODEL_PARALLEL_SIZE']}")
    elif args.set_env:
        scheduler.set_cuda_visible_devices(selected_gpus)
        print("\n环境变量已设置，可以直接启动训练")


if __name__ == '__main__':
    main()
