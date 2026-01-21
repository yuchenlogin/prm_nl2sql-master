"""
TensorBoard训练记录器
记录训练过程中的关键指标参数，替代W&B
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class TensorBoardLogger:
    """TensorBoard日志记录器"""

    def __init__(self, log_dir: str, experiment_name: str = None, port: int = 6006, auto_start: bool = True):
        """
        初始化TensorBoard记录器

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            port: TensorBoard监听端口
            auto_start: 是否自动启动TensorBoard服务器
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置实验名称
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.tensorboard_dir = self.log_dir / "tensorboard" / experiment_name
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard端口配置
        self.port = port
        self.auto_start = auto_start
        self._tb_process = None

        # CSV文件路径
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.csv"

        # 初始化CSV文件
        self._init_csv_file()

        # 指标缓存
        self.step_metrics: List[Dict[str, float]] = []
        self.current_step = 0

        print(f"✅ TensorBoard日志目录: {self.tensorboard_dir}")
        print(f"✅ 指标CSV文件: {self.metrics_file}")
        print(f"✅ TensorBoard端口: {self.port}")

        # 自动启动TensorBoard服务器
        if self.auto_start:
            self.start_tensorboard_server()

    def start_tensorboard_server(self):
        """启动TensorBoard服务器"""
        import subprocess
        import shutil

        # 检查tensorboard是否可用
        if not shutil.which('tensorboard'):
            print("⚠️ TensorBoard未安装，跳过自动启动")
            print("   安装命令: pip install tensorboard")
            return False

        # 检查端口是否已被占用
        if self._is_port_in_use(self.port):
            print(f"⚠️ 端口 {self.port} 已被占用")
            print(f"   可能已经有TensorBoard在运行")
            # 尝试连接查看是否是TensorBoard
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.port))
                if result == 0:
                    print(f"✅ 端口 {self.port} 可用")
                sock.close()
            except:
                pass

            # 如果端口被占用，不启动新的TensorBoard
            return False

        try:
            # 在后台启动TensorBoard
            cmd = [
                'tensorboard',
                '--logdir', str(self.tensorboard_dir),
                '--port', str(self.port),
                '--host', '0.0.0.0',
                '--reload_interval', '30'  # 每30秒重新加载
            ]

            # 启动进程
            self._tb_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False
            )

            # 等待一秒检查进程是否正常启动
            import time
            time.sleep(2)

            if self._tb_process.poll() is None:
                # 进程还在运行
                print(f"\n{'=' * 80}")
                print(f"🚀 TensorBoard服务器已启动")
                print(f"{'=' * 80}")
                print(f"📊 本地访问: http://localhost:{self.port}")
                print(f"📊 远程访问: http://<服务器IP>:{self.port}")
                print(f"📁 日志目录: {self.tensorboard_dir}")
                print(f"{'=' * 80}\n")
                return True
            else:
                # 进程已退出
                stdout, stderr = self._tb_process.communicate()
                print(f"❌ TensorBoard启动失败")
                if stderr:
                    print(f"错误信息: {stderr.decode('utf-8')}")
                return False

        except Exception as e:
            print(f"⚠️ 启动TensorBoard失败: {e}")
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """检查端口是否被占用"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def stop_tensorboard_server(self):
        """停止TensorBoard服务器"""
        if self._tb_process is not None and self._tb_process.poll() is None:
            print(f"正在停止TensorBoard服务器 (端口 {self.port})...")
            self._tb_process.terminate()
            try:
                self._tb_process.wait(timeout=5)
                print("✅ TensorBoard服务器已停止")
            except subprocess.TimeoutExpired:
                print("⚠️ TensorBoard服务器未响应，强制停止")
                self._tb_process.kill()
            self._tb_process = None

    def _init_csv_file(self):
        """初始化CSV文件"""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['step', 'timestamp', 'epoch', 'loss', 'learning_rate',
                               'reward_mean', 'reward_max', 'reward_min', 'reward_std',
                               'gpu_memory_mb', 'train_time_seconds'])

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        记录指标

        Args:
            metrics: 指标字典
            step: 步数，如果不提供则使用内部计数器
        """
        if step is None:
            self.current_step += 1
            step = self.current_step
        else:
            self.current_step = max(self.current_step, step)

        # 添加时间戳和步数
        metrics_with_metadata = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        # 缓存指标
        self.step_metrics.append(metrics_with_metadata)

        # 写入CSV文件
        self._write_to_csv(metrics_with_metadata)

        # 写入TensorBoard格式
        self._write_to_tensorboard(metrics, step)

    def log_training_step(self, step: int, loss: float, learning_rate: float,
                          reward_stats: Dict[str, float] = None, epoch: int = None,
                          gpu_memory_mb: float = None, train_time_seconds: float = None):
        """
        记录训练步骤

        Args:
            step: 步数
            loss: 损失值
            learning_rate: 学习率
            reward_stats: 奖励统计信息
            epoch: 轮数
            gpu_memory_mb: GPU显存使用(MB)
            train_time_seconds: 训练时间(秒)
        """
        metrics = {
            'loss': loss,
            'learning_rate': learning_rate,
        }

        if reward_stats:
            metrics.update({
                'reward_mean': reward_stats.get('mean', 0.0),
                'reward_max': reward_stats.get('max', 0.0),
                'reward_min': reward_stats.get('min', 0.0),
                'reward_std': reward_stats.get('std', 0.0),
            })

        if epoch is not None:
            metrics['epoch'] = epoch

        if gpu_memory_mb is not None:
            metrics['gpu_memory_mb'] = gpu_memory_mb

        if train_time_seconds is not None:
            metrics['train_time_seconds'] = train_time_seconds

        self.log_metrics(metrics, step)

    def _write_to_csv(self, metrics: Dict[str, Any]):
        """将指标写入CSV文件"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # 获取所有字段
            field_names = ['step', 'timestamp', 'epoch', 'loss', 'learning_rate',
                          'reward_mean', 'reward_max', 'reward_min', 'reward_std',
                          'gpu_memory_mb', 'train_time_seconds']

            # 写入行
            row = [metrics.get(name, '') for name in field_names]
            writer.writerow(row)

    def _write_to_tensorboard(self, metrics: Dict[str, float], step: int):
        """
        将指标写入TensorBoard格式
        TensorBoard使用简单的事件文件格式

        Args:
            metrics: 指标字典
            step: 步数
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            if not hasattr(self, 'writer'):
                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

            # 记录每个指标
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'training/{key}', value, step)

            # 刷新写入
            self.writer.flush()

        except ImportError:
            # 如果torch.utils.tensorboard不可用，使用纯Python实现
            self._write_simple_tensorboard(metrics, step)

    def _write_simple_tensorboard(self, metrics: Dict[str, float], step: int):
        """简单的TensorBoard格式写入（不依赖torch）"""
        # 写入到文本文件作为备用方案
        events_file = self.tensorboard_dir / "events.txt"
        with open(events_file, 'a') as f:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{step}\t{key}\t{value}\t{datetime.now().timestamp()}\n")

    def generate_plots(self):
        """生成训练曲线图"""
        if not self.step_metrics:
            print("⚠️ 没有可用的训练数据来生成图表")
            return

        # 准备数据
        steps = [m['step'] for m in self.step_metrics]
        losses = [m.get('loss', 0) for m in self.step_metrics]
        learning_rates = [m.get('learning_rate', 0) for m in self.step_metrics]
        rewards_mean = [m.get('reward_mean', 0) for m in self.step_metrics]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)

        # 损失曲线
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # 学习率曲线
        axes[0, 1].plot(steps, learning_rates, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # 奖励曲线
        if any(r > 0 for r in rewards_mean):
            axes[1, 0].plot(steps, rewards_mean, 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Reward (Mean)')
            axes[1, 0].set_title('Reward Statistics')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No reward data available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # GPU显存使用
        gpu_memory_mb = [m.get('gpu_memory_mb', 0) for m in self.step_metrics]
        if any(m > 0 for m in gpu_memory_mb):
            axes[1, 1].plot(steps, [m/1024 for m in gpu_memory_mb], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('GPU Memory (GB)')
            axes[1, 1].set_title('GPU Memory Usage')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No GPU memory data available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()

        # 保存图表
        plot_file = self.log_dir / f"{self.experiment_name}_training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 训练曲线图已保存: {plot_file}")

        # 保存为JSON格式
        json_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(self.step_metrics, f, indent=2)
        print(f"✅ 指标JSON文件已保存: {json_file}")

    def finish(self):
        """完成记录"""
        if hasattr(self, 'writer'):
            self.writer.close()

        # 生成最终图表
        self.generate_plots()

        print("\n" + "=" * 80)
        print("训练日志已保存")
        print("=" * 80)
        print(f"📊 TensorBoard目录: {self.tensorboard_dir}")
        print(f"📊 本地访问: http://localhost:{self.port}")
        print(f"📊 远程访问: http://<服务器IP>:{self.port}")
        print(f"📈 查看训练曲线: tensorboard --logdir={self.tensorboard_dir} --port {self.port}")
        print(f"📊 CSV文件: {self.metrics_file}")
        print(f"💡 提示: 训练过程中TensorBoard已自动启动，无需手动启动")
        print("=" * 80)


class MetricsTracker:
    """简单的指标跟踪器"""

    def __init__(self):
        self.metrics_history = {}

    def update(self, metric_name: str, value: float, step: int):
        """更新指标"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []

        self.metrics_history[metric_name].append({
            'step': step,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self, metric_name: str) -> Dict:
        """获取指标摘要"""
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return {}

        values = [m['value'] for m in self.metrics_history[metric_name]]

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1] if values else None,
            'count': len(values)
        }
