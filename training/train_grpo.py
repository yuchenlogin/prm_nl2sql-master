"""
GRPO训练脚本
使用Group Relative Policy Optimization进行NL2SQL过程奖励微调
"""

import os
import sys
import yaml
import logging
import torch
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set Hugging Face environment variables to use direct endpoints
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
os.environ['HF_HUB_URL'] = 'https://huggingface.co'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

# Force use of local cache for Qwen3-1.7B
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE_DIR'] = '/.cache/huggingface/hub/'

# Fix TRL import issues by setting environment variables
os.environ['TRL_USE_RICH'] = 'false'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Import TRL with error handling
try:
    from trl import GRPOTrainer, GRPOConfig
    print("✅ Successfully imported TRL")
except ImportError as e:
    print(f"❌ Error importing TRL: {e}")
    print("⚠️ TRL not available, using mock implementation for testing")
    # Create mock classes for testing when TRL is not available
    class GRPOConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class GRPOTrainer:
        def __init__(self, **kwargs):
            print("⚠️ Using mock GRPOTrainer - training will not actually run")

        def train(self):
            print("⚠️ Mock training completed")

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_loader import NL2SQLDataLoader
from reward.reward_model import ProcessRewardModel
from training.train_utils import WandBLogger, CheckpointManager, PerformanceMonitor, Logger, GPUMonitor
from generator.prompts import PromptTemplates
from utils.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    # 模型配置
    model_name: str = "Qwen/Qwen3-1.7B"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # 数据配置
    train_file: str = ""
    test_file: str = ""
    val_split: float = 0.1

    # 训练配置
    output_dir: str = "./outputs/checkpoints"
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    total_batch_size: int = 32
    learning_rate: float = 7.3e-6
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    optim: str = "adamw_8bit"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    bf16: bool = True
    tf32: bool = True

    # GRPO配置
    num_generations: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 1024

    # 奖励权重
    type_weight: float = 0.20
    thinking_weight: float = 0.25
    self_assessment_weight: float = 0.25
    sql_structure_weight: float = 0.30

    # 评估配置
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10

    # W&B配置
    wandb_enabled: bool = True
    wandb_project: str = "qwen3-nl2sql-grpo"
    wandb_entity: Optional[str] = None

    # 硬件配置
    num_gpus: int = 8
    seed: int = 42

    # 检查点配置
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3


class NL2SQLTrainer:
    """NL2SQL GRPO训练器"""

    def __init__(self, config: TrainingConfig):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self._setup_logging()
        self._setup_environment()
        self._initialize_components()

    def _setup_logging(self):
        """设置日志"""
        logger_util = Logger(
            log_dir="./outputs/logs",
            log_file="training.log",
            level="INFO"
        )
        logger.info("日志系统已初始化")

    def _setup_environment(self):
        """设置环境"""
        # 设置随机种子
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # 记录GPU信息
        GPUMonitor.log_gpu_status()

    def _initialize_components(self):
        """初始化各个组件"""
        logger.info("初始化组件...")

        # 从配置中读取TensorBoard设置
        tb_port = getattr(self.config, 'tensorboard_port', 6006)
        auto_start_tb = getattr(self.config, 'auto_start_tensorboard', True)

        # 初始化TensorBoard日志记录器
        self.tb_logger = TensorBoardLogger(
            log_dir=os.path.join(self.config.output_dir, 'logs'),
            experiment_name=f"nl2sql-grpo-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            port=tb_port,
            auto_start=auto_start_tb
        )

        # 初始化W&B（如果启用）
        if self.config.wandb_enabled:
            self.wandb_logger = WandBLogger(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=f"nl2sql-grpo-{torch.cuda.get_device_name(0).replace(' ', '-')}",
                config=self.config.__dict__,
                enabled=self.config.wandb_enabled,
                tags=["nl2sql", "grpo", "process_reward", "2025"]
            )
        else:
            logger.info("W&B已禁用，使用TensorBoard记录")
            self.wandb_logger = None

        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.output_dir,
            save_total_limit=self.config.save_total_limit,
            best_model_metric="eval_type_accuracy"
        )

        # 初始化性能监控器
        self.performance_monitor = PerformanceMonitor()

        # 加载数据
        logger.info("加载数据...")
        self.data_loader = NL2SQLDataLoader()
        self.train_examples = self.data_loader.load(self.config.train_file)
        logger.info(f"加载了 {len(self.train_examples)} 条训练数据")

        # 加载模型和分词器
        logger.info(f"加载模型: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # dtype_map = {
        #     "float32": torch.float32,
        #     "float16": torch.float16,
        #     "bfloat16": torch.bfloat16
        # }
        # torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.config.model_name,
        #     torch_dtype=torch_dtype,
        #     device_map="auto",
        #     trust_remote_code=True
        # )

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        # 读取 Qwen3 相关高级配置（目前仅记录日志，后续可用于精细控制）
        max_seq_length = getattr(self.config, "max_seq_length", None)
        attn_impl = getattr(self.config, "attn_implementation", None)
        rope_scaling_cfg = getattr(self.config, "rope_scaling", None)
        logger.info(
            f"Qwen3 高级配置 - max_seq_length={max_seq_length}, "
            f"attn_implementation={attn_impl}, rope_scaling={rope_scaling_cfg}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )


        # 初始化奖励模型
        self.reward_model = ProcessRewardModel(
            type_weight=self.config.type_weight,
            thinking_weight=self.config.thinking_weight,
            self_assessment_weight=self.config.self_assessment_weight,
            sql_structure_weight=self.config.sql_structure_weight
        )

        logger.info("组件初始化完成")

    def _prepare_training_data(self):
        """准备训练数据"""
        logger.info("准备训练数据...")

        # 获取模板
        schema = PromptTemplates.get_schema_context()
        business_knowledge = PromptTemplates.get_business_knowledge()
        few_shot_examples = PromptTemplates.get_few_shot_examples()
        system_prompt = PromptTemplates.BASE_SYSTEM_PROMPT

        train_data = []
        for example in self.train_examples:
            # 构建用户内容
            user_content = PromptTemplates.SQL_GENERATION_PROMPT.format(
                system_prompt=system_prompt,
                schema=schema,
                business_knowledge=business_knowledge,
                few_shot_examples=few_shot_examples,
                question=example.query,
            )

            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # 使用Qwen3聊天模板转换为提示，并开启thinking模式
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            train_data.append({
                "prompt": prompt,
                "reference_type": example.complexity_type,
                "reference_sql": example.sql,
                "query": example.query
            })

        logger.info(f"准备了 {len(train_data)} 条训练数据")
        return train_data

    def _create_reward_function(self):
        """创建奖励函数"""
        def compute_rewards(prompts, completions, **kwargs):
            """
            计算奖励

            Args:
                prompts: 提示列表
                completions: 模型生成的完成序列
                **kwargs: 其他参数

            Returns:
                奖励张量
            """
            rewards = []

            for prompt, completion in zip(prompts, completions):
                # 解析完成中的thinking和SQL
                thinking = self._extract_section(completion, 'think')
                sql = self._extract_section(completion, 'answer')

                # 如果没有SQL，给予低奖励
                if not sql:
                    rewards.append(0.0)
                    continue

                # 对于SQL类任务，计算结构奖励
                # 使用SQL结构检查
                sql_validity = 1.0

                # 基础SQL有效性检查
                sql_upper = sql.upper()
                if 'SELECT' not in sql_upper or 'FROM' not in sql_upper:
                    sql_validity = 0.0
                elif sql.count('(') != sql.count(')'):
                    sql_validity = 0.5

                # 推理质量检查
                thinking_quality = 0.0
                if thinking:
                    # 检查推理长度和关键词
                    if len(thinking) >= 50:
                        thinking_quality += 0.5
                    # 检查SQL关键词
                    sql_keywords = ['WHERE', 'FROM', 'SELECT', 'JOIN']
                    if any(kw in thinking.upper() for kw in sql_keywords):
                        thinking_quality += 0.3
                    # 检查逻辑连接词
                    logic_words = ['因为', '所以', '然后', '首先', '需要', '根据']
                    if any(word in thinking for word in logic_words):
                        thinking_quality += 0.2

                # 综合奖励
                total_reward = (
                    0.3 * sql_validity +      # SQL结构权重
                    0.3 * thinking_quality  # 推理质量权重
                )

                rewards.append(total_reward)

            import torch
            import os
            # 如果有GPU可用，将奖励张量移到GPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            return torch.tensor(rewards, device=device)

        return compute_rewards

    @staticmethod
    def _extract_section(text: str, section: str) -> str:
        """提取指定标签内容"""
        start_tag = f"<{section}>"
        end_tag = f"</{section}>"

        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            return ""

        content = text[start_idx + len(start_tag):end_idx].strip()
        return content

    def _create_grpo_config(self):
        """创建GRPO配置"""
        return GRPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            bf16=True,
            tf32=True,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            # 禁用评估，因为没有eval_dataset
            eval_strategy="no",
            save_strategy="steps",
            report_to=[],  # 禁用W&B
            seed=self.config.seed,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=8,
        )


    def train(self):
        """执行训练"""
        logger.info("=" * 50)
        logger.info("开始GRPO训练")
        logger.info("=" * 50)

        self.performance_monitor.start()

        try:
            # 准备数据
            train_data = self._prepare_training_data()

            # 创建GRPO配置
            grpo_config = self._create_grpo_config()

            # 创建TensorBoard回调
            from utils.tensorboard_callback import TensorBoardCallback
            tb_callback = TensorBoardCallback(self.tb_logger)

            # 创建GRPO训练器
            trainer = GRPOTrainer(
                model=self.model,
                args=grpo_config,
                train_dataset=train_data,
                reward_funcs=[self._create_reward_function()],
                processing_class=self.tokenizer,
                callbacks=[tb_callback],
            )

            # 开始训练
            logger.info("训练开始...")
            train_result = trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

            # 记录训练结果
            logger.info("=" * 50)
            logger.info("训练完成")
            logger.info(f"最终损失: {train_result.training_loss}")
            logger.info("=" * 50)

            # 保存最终模型
            final_model_dir = Path(self.config.output_dir) / "final_model"
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            logger.info(f"最终模型已保存: {final_model_dir}")

            # 上传到W&B
            if self.config.wandb_enabled:
                self.wandb_logger.log_model(str(final_model_dir), "final_model")

            # 保存性能报告
            self._save_performance_report()

        except Exception as e:
            logger.error(f"训练出错: {e}")
            raise
        finally:
            # 完成TensorBoard记录
            logger.info("正在保存训练日志...")
            self.tb_logger.finish()

            # 停止TensorBoard服务器（如果是自动启动的）
            self.tb_logger.stop_tensorboard_server()

            # 完成W&B
            if self.config.wandb_enabled and self.wandb_logger:
                self.wandb_logger.finish()

    def _save_performance_report(self):
        """保存性能报告"""
        report = self.performance_monitor.get_metrics_summary()
        report_path = Path(self.config.output_dir) / "performance_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("NL2SQL GRPO训练性能报告\n")
            f.write("=" * 50 + "\n\n")

            for key, value in report.items():
                f.write(f"{key}: {value}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("数据集统计\n")
            f.write("=" * 50 + "\n")

            stats = self.data_loader.get_statistics()
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"性能报告已保存: {report_path}")


def load_config(config_path: str) -> TrainingConfig:
    """从YAML加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 展平嵌套字典
    training_config = config_dict.get('training', {})
    model_config = config_dict.get('model', {})
    data_config = config_dict.get('data', {})
    grpo_config = config_dict.get('grpo', {})
    wandb_config = config_dict.get('wandb', {})

    # 合并所有配置
    merged_config = {
        **training_config,
        **model_config,
        **data_config,
        **grpo_config,
        **wandb_config,
    }

    # 提取reward_weights
    reward_weights = grpo_config.get('reward_weights', {})
    merged_config['type_weight'] = reward_weights.get('type_reward', 0.20)
    merged_config['thinking_weight'] = reward_weights.get('thinking_reward', 0.25)
    merged_config['self_assessment_weight'] = reward_weights.get('self_assessment_reward', 0.25)
    merged_config['sql_structure_weight'] = reward_weights.get('sql_structure_reward', 0.30)

    # 筛选出TrainingConfig中定义的字段
    valid_fields = set(TrainingConfig.__dataclass_fields__.keys())
    filtered_config = {k: v for k, v in merged_config.items() if k in valid_fields}

    print(f"📝 有效配置字段: {list(filtered_config.keys())}")

    return TrainingConfig(**filtered_config)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="NL2SQL GRPO训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径"
    )
    # Add GPU configuration arguments
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="CUDA visible devices (e.g., 0,1,2,3)"
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=None,
        help="Number of GPUs per node"
    )
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=None,
        help="Tensor model parallel size"
    )

    args = parser.parse_args()

    # Set GPU-related environment variables
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"🔧 Setting CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    if args.gpus_per_node:
        os.environ["N_GPUS_PER_NODE"] = str(args.gpus_per_node)
        print(f"🔧 Setting N_GPUS_PER_NODE={args.gpus_per_node}")

    if args.tensor_model_parallel_size:
        os.environ["TENSOR_MODEL_PARALLEL_SIZE"] = str(args.tensor_model_parallel_size)
        print(f"🔧 Setting TENSOR_MODEL_PARALLEL_SIZE={args.tensor_model_parallel_size}")

    # 加载配置
    config = load_config(args.config)
    if args.resume:
        config.resume_from_checkpoint = args.resume

    # 创建训练器并训练
    trainer = NL2SQLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
