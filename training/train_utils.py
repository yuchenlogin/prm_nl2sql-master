"""
训练工具模块
日志、检查点、W&B集成等辅助功能
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import wandb
import torch

logger = logging.getLogger(__name__)


class WandBLogger:
    """Weights & Biases日志记录器"""

    def __init__(self,
                 project: str = "qwen3-nl2sql-grpo",
                 entity: Optional[str] = None,
                 name: Optional[str] = None,
                 config: Optional[Dict] = None,
                 enabled: bool = True,
                 tags: Optional[list] = None):
        """
        初始化W&B日志记录器

        Args:
            project: W&B项目名称
            entity: W&B实体（用户名或团队名）
            name: 实验运行名称
            config: 配置字典
            enabled: 是否启用W&B
            tags: 实验标签
        """
        self.enabled = enabled
        self.project = project
        self.entity = entity

        if enabled:
            try:
                wandb.init(
                    project=project,
                    entity=entity,
                    name=name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config=config,
                    tags=tags or []
                )
                logger.info("W&B已初始化")
            except Exception as e:
                logger.warning(f"W&B初始化失败: {e}")
                self.enabled = False

    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """记录指标"""
        if self.enabled:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"W&B日志记录失败: {e}")

    def log_reward_breakdown(self, rewards: Dict, step: int):
        """记录奖励分解"""
        if self.enabled:
            try:
                wandb.log({
                    'reward/total': rewards.get('total_reward', 0),
                    'reward/type': rewards.get('type_reward', 0),
                    'reward/thinking': rewards.get('thinking_reward', 0),
                    'reward/self_assessment': rewards.get('self_assessment_reward', 0),
                    'reward/sql_structure': rewards.get('sql_structure_reward', 0),
                }, step=step)
            except Exception as e:
                logger.warning(f"W&B奖励日志记录失败: {e}")

    def log_model(self, model_path: str, name: str = "model"):
        """上传模型到W&B"""
        if self.enabled:
            try:
                artifact = wandb.Artifact(name, type='model')
                artifact.add_dir(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"模型已上传到W&B: {model_path}")
            except Exception as e:
                logger.warning(f"W&B模型上传失败: {e}")

    def finish(self):
        """完成W&B记录"""
        if self.enabled:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B完成失败: {e}")


class CheckpointManager:
    """检查点管理器"""

    def __init__(self,
                 output_dir: str = "./outputs/checkpoints",
                 save_total_limit: int = 3,
                 best_model_metric: Optional[str] = None):
        """
        初始化检查点管理器

        Args:
            output_dir: 输出目录
            save_total_limit: 保存的最大检查点数
            best_model_metric: 用于选择最佳模型的指标
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.best_model_metric = best_model_metric
        self.best_metric_value = None
        self.checkpoint_list = []

    def save_checkpoint(self,
                       model,
                       tokenizer,
                       optimizer,
                       step: int,
                       metrics: Optional[Dict] = None):
        """
        保存检查点

        Args:
            model: 模型
            tokenizer: 分词器
            optimizer: 优化器
            step: 当前步数
            metrics: 指标字典
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型和分词器
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # 保存优化器状态
        optimizer_state = {
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'metrics': metrics or {}
        }
        with open(checkpoint_dir / 'trainer_state.json', 'w') as f:
            json.dump({
                'global_step': step,
                'best_metric': self.best_metric_value,
                'metrics': metrics or {}
            }, f, indent=2)

        self.checkpoint_list.append(checkpoint_dir)
        logger.info(f"检查点已保存: {checkpoint_dir}")

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        # 检查是否是最佳模型
        if metrics and self.best_model_metric and self.best_model_metric in metrics:
            metric_value = metrics[self.best_model_metric]
            if self.best_metric_value is None or metric_value > self.best_metric_value:
                self.best_metric_value = metric_value
                self._save_best_model(checkpoint_dir)

    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        if len(self.checkpoint_list) > self.save_total_limit:
            old_checkpoint = self.checkpoint_list.pop(0)
            try:
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"已删除旧检查点: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"删除旧检查点失败: {e}")

    def _save_best_model(self, checkpoint_dir: Path):
        """保存最佳模型"""
        best_model_dir = self.output_dir / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        try:
            import shutil
            # 复制最佳模型
            for file in checkpoint_dir.glob("*"):
                if file.is_file():
                    shutil.copy2(file, best_model_dir)
                elif file.is_dir():
                    shutil.copytree(file, best_model_dir / file.name, dirs_exist_ok=True)
            logger.info(f"最佳模型已保存: {best_model_dir}")
        except Exception as e:
            logger.warning(f"保存最佳模型失败: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            logger.error(f"检查点不存在: {checkpoint_path}")
            return None

        return checkpoint_dir


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        """初始化性能监控器"""
        self.metrics_history = []
        self.training_start_time = None

    def start(self):
        """开始计时"""
        self.training_start_time = datetime.now()

    def record_metrics(self, metrics: Dict, step: int):
        """记录指标"""
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['step'] = step
        if self.training_start_time:
            elapsed = (datetime.now() - self.training_start_time).total_seconds()
            metrics['elapsed_seconds'] = elapsed
        self.metrics_history.append(metrics)

    def get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}

        summary = {
            'total_steps': len(self.metrics_history),
            'elapsed_time': None
        }

        if self.training_start_time:
            elapsed = (datetime.now() - self.training_start_time).total_seconds()
            summary['elapsed_time'] = f"{elapsed/3600:.2f} hours"
            summary['avg_time_per_step'] = f"{elapsed/len(self.metrics_history):.2f}s"

        # 计算奖励统计
        rewards = [m.get('loss', 0) for m in self.metrics_history if 'loss' in m]
        if rewards:
            summary['avg_loss'] = sum(rewards) / len(rewards)
            summary['min_loss'] = min(rewards)
            summary['max_loss'] = max(rewards)

        return summary

    def save_history(self, output_path: str):
        """保存指标历史"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"指标历史已保存: {output_file}")


class Logger:
    """统一日志管理器"""

    def __init__(self,
                 log_dir: str = "./outputs/logs",
                 log_file: str = "training.log",
                 level: str = "INFO"):
        """
        初始化日志管理器

        Args:
            log_dir: 日志目录
            log_file: 日志文件名
            level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.log_dir / log_file
        self._setup_logger(log_path, level)

    def _setup_logger(self, log_path: Path, level: str):
        """设置日志记录器"""
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, level))

        # 文件处理器
        fh = logging.FileHandler(log_path)
        fh.setLevel(getattr(logging, level))

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level))

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(fh)
        logger.addHandler(ch)


class GPUMonitor:
    """GPU监控器"""

    @staticmethod
    def get_gpu_info() -> Dict:
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return {'available': False}

        info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'devices': []
        }

        for i in range(torch.cuda.device_count()):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                'current_memory_gb': torch.cuda.memory_allocated(i) / 1e9
            }
            info['devices'].append(device_info)

        return info

    @staticmethod
    def log_gpu_status():
        """记录GPU状态"""
        info = GPUMonitor.get_gpu_info()
        if info['available']:
            for device in info['devices']:
                logger.info(
                    f"GPU {device['id']} ({device['name']}): "
                    f"{device['current_memory_gb']:.2f}GB / {device['total_memory_gb']:.2f}GB"
                )
        else:
            logger.warning("CUDA不可用")
