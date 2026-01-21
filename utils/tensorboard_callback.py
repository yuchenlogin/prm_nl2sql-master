"""
TensorBoard训练回调
集成到Trainer中用于记录训练指标
"""

import torch
import logging
from typing import Dict, Any
from transformers import TrainerCallback
from utils.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


class TensorBoardCallback(TrainerCallback):
    """TensorBoard回调类，用于记录训练指标"""

    def __init__(self, tensorboard_logger: TensorBoardLogger):
        """
        初始化TensorBoard回调

        Args:
            tensorboard_logger: TensorBoard日志记录器实例
        """
        self.tb_logger = tensorboard_logger
        self.log_history = []

    def on_log(self, args, state, control, logs: Dict[str, float] = None, **kwargs):
        """
        在每次日志记录时调用

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志字典
        """
        if logs is None:
            return

        self.log_history.append(logs.copy())

        # 提取关键指标
        step = state.global_step
        epoch = state.epoch

        metrics = {
            'epoch': epoch
        }

        # 提取损失相关指标
        if 'loss' in logs:
            metrics['loss'] = float(logs['loss'])
        if 'train_loss' in logs:
            metrics['loss'] = float(logs['train_loss'])

        # 提取学习率
        if 'learning_rate' in logs:
            metrics['learning_rate'] = float(logs['learning_rate'])
        else:
            # 从优化器获取学习率
            if kwargs.get('optimizer') is not None:
                optimizer = kwargs['optimizer']
                if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                    metrics['learning_rate'] = optimizer.param_groups[0].get('lr', 0.001)

        # 提取奖励相关指标
        reward_keys = ['reward', 'reward/mean', 'reward_mean', 'avg_reward']
        for key in reward_keys:
            if key in logs:
                metrics['reward_mean'] = float(logs[key])
                break

        # 提取梯度相关指标
        if 'grad_norm' in logs:
            metrics['grad_norm'] = float(logs['grad_norm'])

        # 提取时间相关指标
        if 'train_runtime' in logs:
            metrics['train_time_seconds'] = float(logs['train_runtime'])

        # 记录到TensorBoard
        if metrics:
            self.tb_logger.log_metrics(metrics, step)

    def on_evaluate(self, args, state, control, metrics: Dict[str, float] = None, **kwargs):
        """
        在评估时调用

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
            metrics: 评估指标
        """
        if metrics is None:
            return

        step = state.global_step

        # 添加eval前缀
        eval_metrics = {f'eval_{k}': float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

        self.tb_logger.log_metrics(eval_metrics, step)

    def on_step_end(self, args, state, control, **kwargs):
        """
        在训练步骤结束时调用

        Args:
            args: 训练参数
            state: 训练状态
            control: 训练控制
        """
        # 记录GPU显存使用
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self.tb_logger.log_metrics({
                    'gpu_memory_mb': gpu_memory_mb
                }, state.global_step)
            except:
                pass
