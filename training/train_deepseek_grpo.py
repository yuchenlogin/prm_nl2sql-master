"""
DeepSeek GRPO训练脚本

基于DeepSeek-Math-V2的自验证机制，集成到现有的GRPO训练框架
实现：过程奖励微调 + 三层验证 + 迭代优化
"""

import os
import sys
import warnings
import yaml
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import NL2SQLDataLoader
from deepseek_sql import DeepSeekNL2SQL
from training.train_utils import WandBLogger, CheckpointManager, PerformanceMonitor, Logger, GPUMonitor
from utils.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekTrainingConfig:
    """DeepSeek GRPO训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen3-1.7B"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True

    # 数据配置
    train_file: str = ""
    test_file: str = ""
    val_split: float = 0.1

    # 训练配置
    output_dir: str = "./outputs/deepseek_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # 由于DeepSeek开销大，减少批次大小
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # 增加梯度累积来补偿
    learning_rate: float = 5e-6  # 降低学习率
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    bf16: bool = True
    tf32: bool = True

    # DeepSeek配置
    deepseek_max_rounds: int = 3
    deepseek_n_generations_per_round: int = 2
    deepseek_n_verifications_per_generation: int = 2
    deepseek_process_reward_weight: float = 0.7  # 过程奖励权重
    deepseek_final_reward_weight: float = 0.3   # 最终结果奖励权重

    # VLM验证器配置
    vlm_enabled: bool = False  # 是否启用VLM验证器
    vlm_model_path: Optional[str] = None  # VLM模型路径
    vlm_verification_weight: float = 0.8  # VLM验证权重

    # GRPO配置
    num_generations: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 1024

    # 评估和保存配置
    eval_steps: int = 200  # 减少评估频率
    save_steps: int = 200  # 减少保存频率
    save_total_limit: int = 3
    logging_steps: int = 20
    logging_dir: str = "./outputs/logs"

    # TensorBoard 配置
    tensorboard_port: int = 6007
    auto_start_tensorboard: bool = True
    tensorboard_log_dir: str = "./outputs/deepseek_checkpoints/logs"

    # W&B配置
    wandb_enabled: bool = True
    wandb_project: str = "qwen3-nl2sql-deepseek-grpo"
    wandb_entity: Optional[str] = None
    wandb_name: str = "deepseek_process_reward_training"

    # 硬件配置
    num_gpus: int = 8
    seed: int = 42

    # 检查点配置
    resume_from_checkpoint: Optional[str] = None


class DeepSeekGRPOTrainer:
    """DeepSeek GRPO训练器"""

    def __init__(self, config: DeepSeekTrainingConfig):
        """
        初始化DeepSeek GRPO训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self._setup_logging()
        self._setup_environment()
        self._initialize_components()

    def _setup_logging(self):
        """设置日志"""
        import warnings

        # 过滤警告
        warnings.filterwarnings("ignore")
        os.environ['PYTHONWARNINGS'] = 'ignore'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'./outputs/logs/deepseek_training_{self.config.seed}.log'),
                logging.StreamHandler()
            ]
        )

        # 抑制urllib3详细日志
        logging.getLogger("urllib3").setLevel(logging.ERROR)

    def _setup_environment(self):
        """设置环境"""
        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path("./outputs/logs").mkdir(parents=True, exist_ok=True)
        Path("./outputs/deepseek_proof_pool").mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _initialize_components(self):
        """初始化组件"""
        logger.info("Initializing DeepSeek GRPO components...")

        # 初始化TensorBoard日志记录器
        self.tb_logger = TensorBoardLogger(
            log_dir=self.config.tensorboard_log_dir,
            experiment_name=f"deepseek-grpo-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            port=self.config.tensorboard_port,
            auto_start=self.config.auto_start_tensorboard
        )

        # 初始化W&B
        if self.config.wandb_enabled:
            self.wandb_logger = WandBLogger(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_name,
                config=self.config
            )
        else:
            self.wandb_logger = None

        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.output_dir,
            save_total_limit=self.config.save_total_limit
        )

        # 初始化性能监控器
        self.performance_monitor = PerformanceMonitor()

        # 加载数据
        logger.info("Loading training data...")
        self.data_loader = NL2SQLDataLoader()
        self.train_examples = self.data_loader.load(self.config.train_file)

        # Split into train and validation
        val_size = int(len(self.train_examples) * self.config.val_split)
        self.train_dataset = self.train_examples[:-val_size]
        self.val_dataset = self.train_examples[-val_size:]

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")

        # 加载模型和分词器
        logger.info(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # 初始化DeepSeek NL2SQL系统
        logger.info("Initializing DeepSeek NL2SQL system...")
        vlm_model_path = None
        if hasattr(self.config, 'vlm_enabled') and self.config.vlm_enabled:
            vlm_model_path = getattr(self.config, 'vlm_model_path', None)
            logger.info(f"VLM verification enabled with model: {vlm_model_path}")

        self.deepseek_system = DeepSeekNL2SQL(
            model_name=self.config.model_name,
            pool_dir="./outputs/deepseek_proof_pool",
            vlm_model_path=vlm_model_path
        )

        logger.info("Component initialization completed")

    def train(self):
        """开始训练"""
        logger.info("=" * 50)
        logger.info("开始 DeepSeek GRPO 训练")
        logger.info("=" * 50)

        logger.info("Starting DeepSeek GRPO training...")
        logger.info(f"Training data size: {len(self.train_dataset)}")
        logger.info(f"Validation data size: {len(self.val_dataset)}")
        logger.info(f"DeepSeek max rounds: {self.config.deepseek_max_rounds}")

        self.performance_monitor.start()

        try:
            # 准备训练数据
            train_data = self._prepare_training_data()

            # 配置GRPO
            grpo_config = GRPOConfig(
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
                num_generations=self.config.num_generations,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_new_tokens=self.config.max_new_tokens,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                logging_steps=self.config.logging_steps,
                seed=self.config.seed,
                bf16=self.config.bf16,
                tf32=self.config.tf32,
                report_to=[],
                remove_unused_columns=False,
            )

            # 创建TensorBoard回调
            from utils.tensorboard_callback import TensorBoardCallback
            tb_callback = TensorBoardCallback(self.tb_logger)

            # 创建GRPO训练器
            grpo_trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                args=grpo_config,
                train_dataset=train_data,
                eval_dataset=self.val_dataset,
                reward_funcs=[self._deepseek_reward_function],
                callbacks=[tb_callback],
            )

            # 开始训练
            logger.info("训练开始...")
            train_result = grpo_trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

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
            if self.config.wandb_enabled and self.wandb_logger:
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

        logger.info("DeepSeek GRPO training completed")

    def _save_performance_report(self):
        """保存性能报告"""
        report = self.performance_monitor.get_metrics_summary()
        report_path = Path(self.config.output_dir) / "performance_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("NL2SQL DeepSeek GRPO训练性能报告\n")
            f.write("=" * 50 + "\n\n")

            for key, value in report.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"性能报告已保存: {report_path}")

    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """准备训练数据（并行处理）"""
        logger.info("Preparing training data with DeepSeek processing (parallel mode)...")

        training_data = []

        # 使用并行处理加速数据准备
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        # 限制并行进程数 = CPU核心数或4，取较小值
        num_workers = min(os.cpu_count() or 4, 4)
        logger.info(f"Using {num_workers} workers for parallel processing")

        # 准备所有样本的参数
        sample_params = []
        for i, example in enumerate(self.train_dataset[:100]):  # 限制100个样本
            query = example.query
            response = example.response
            reference_type = example.complexity_type
            reference_sql = example.sql

            # 从query中提取组件
            schema, knowledge, examples = self._parse_query_components(query)
            actual_query = self._extract_question_from_query(query)

            if not actual_query or not schema:
                continue

            sample_params.append({
                'idx': i,
                'query': actual_query,
                'schema': schema,
                'knowledge': knowledge,
                'examples': examples,
                'reference_type': reference_type,
                'reference_sql': reference_sql,
                'model_name': self.config.model_name,
                'pool_dir': "./outputs/deepseek_proof_pool",
                'vlm_model_path': self.config.vlm_model_path if hasattr(self.config, 'vlm_enabled') and self.config.vlm_enabled else None
            })

        logger.info(f"Processing {len(sample_params)} samples in parallel...")

        # 并行处理样本
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(_process_single_sample, params): params['idx']
                for params in sample_params
            }

            # 收集结果
            processed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result and result.get('success', False):
                        training_data.append(result['data'])
                        processed_count += 1

                    # 进度报告
                    if (processed_count + len(training_data)) % 10 == 0:
                        logger.info(f"Completed {(processed_count + len(training_data))}/{len(sample_params)} samples...")

                except Exception as e:
                    logger.debug(f"Error processing sample {idx}: {e}")

        logger.info(f"Training data preparation completed: {len(training_data)} samples ready from {len(sample_params)} attempts")
        return training_data

    def _deepseek_reward_function(self, prompts: List[str], responses: List[str],
                                 references: List[str], **kwargs) -> List[float]:
        """
        DeepSeek过程奖励函数

        Args:
            prompts: 提示列表
            responses: 生成的响应列表
            references: 参考答案列表

        Returns:
            奖励分数列表
        """
        rewards = []

        for i, (prompt, response, reference) in enumerate(zip(prompts, responses, references)):
            try:
                # 解析组件
                query, schema, knowledge, examples = self._parse_training_prompt(prompt)

                if not all([query, schema, response]):
                    rewards.append(0.0)
                    continue

                # 解析生成的响应
                generated_sql = self._extract_sql_from_response(response)
                thinking = self._extract_thinking_from_response(response)
                self_eval = self._extract_self_eval_from_response(response)

                if not generated_sql:
                    rewards.append(0.0)
                    continue

                # 使用DeepSeek系统进行评分
                deepseek_result = self.deepseek_system.process_query(
                    query=query,
                    schema=schema,
                    knowledge=knowledge,
                    examples=examples,
                    problem_idx=f"eval_{i}"
                )

                if deepseek_result.get('success', False):
                    # 获取过程奖励
                    process_reward = deepseek_result.get('process_reward', {})
                    total_process_reward = process_reward.get('total_process_reward', 0.0)

                    # 获取最终结果分数
                    best_score = deepseek_result.get('best_score', 0.0)

                    # 组合奖励
                    combined_reward = (
                        total_process_reward * self.config.deepseek_process_reward_weight +
                        best_score * self.config.deepseek_final_reward_weight
                    )

                    rewards.append(combined_reward)
                else:
                    rewards.append(0.0)

            except Exception as e:
                logger.error(f"Error calculating reward for sample {i}: {e}")
                rewards.append(0.0)

        # 记录奖励统计
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            logger.debug(f"Average reward for batch: {avg_reward:.4f}")

        return rewards

    def _parse_response_components(self, response: str) -> tuple:
        """解析响应组件"""
        # 从响应中提取schema, knowledge等
        # 这里需要根据实际的数据格式进行解析
        schema = ""
        knowledge = ""
        examples = ""

        try:
            # 简单的解析逻辑，需要根据实际数据格式调整
            lines = response.split('\n')
            current_section = None
            section_content = []

            for line in lines:
                if line.startswith('--- 2.schema kg'):
                    current_section = 'schema'
                elif line.startswith('--- 3.knowledge graph'):
                    current_section = 'knowledge'
                elif line.startswith('--- 6.few shot'):
                    current_section = 'examples'
                elif line.startswith('---'):
                    # 结束当前section
                    if current_section == 'schema':
                        schema = '\n'.join(section_content)
                    elif current_section == 'knowledge':
                        knowledge = '\n'.join(section_content)
                    elif current_section == 'examples':
                        examples = '\n'.join(section_content)
                    current_section = None
                    section_content = []
                elif current_section:
                    section_content.append(line)

        except Exception as e:
            logger.warning(f"Error parsing response components: {e}")

        return schema, knowledge, examples

    def _parse_query_components(self, query: str) -> tuple:
        """从query中提取schema, knowledge, examples组件"""
        schema = ""
        knowledge = ""
        examples = ""

        try:
            lines = query.split('\n')
            current_section = None
            section_content = []

            for line in lines:
                # 先检查新的section标记，如果是新section，先保存当前section
                if '--- 2.' in line and 'schema' in line.lower():
                    # 保存前一个section（如果有）
                    if current_section == 'schema':
                        schema = '\n'.join(section_content)
                    elif current_section == 'knowledge':
                        knowledge = '\n'.join(section_content)
                    elif current_section == 'examples':
                        examples = '\n'.join(section_content)

                    current_section = 'schema'
                    section_content = []
                elif '--- 3.' in line and 'knowledge' in line.lower():
                    # 保存前一个section
                    if current_section == 'schema':
                        schema = '\n'.join(section_content)
                    elif current_section == 'knowledge':
                        knowledge = '\n'.join(section_content)
                    elif current_section == 'examples':
                        examples = '\n'.join(section_content)

                    current_section = 'knowledge'
                    section_content = []
                elif '--- 6.' in line and 'few' in line.lower():
                    # 保存前一个section
                    if current_section == 'schema':
                        schema = '\n'.join(section_content)
                    elif current_section == 'knowledge':
                        knowledge = '\n'.join(section_content)
                    elif current_section == 'examples':
                        examples = '\n'.join(section_content)

                    current_section = 'examples'
                    section_content = []
                elif line.startswith('---'):
                    # 其他section标记，保存当前section
                    if current_section == 'schema':
                        schema = '\n'.join(section_content)
                    elif current_section == 'knowledge':
                        knowledge = '\n'.join(section_content)
                    elif current_section == 'examples':
                        examples = '\n'.join(section_content)
                    current_section = None
                    section_content = []
                elif current_section:
                    section_content.append(line)

            # 处理最后一个section
            if current_section == 'schema':
                schema = '\n'.join(section_content)
            elif current_section == 'knowledge':
                knowledge = '\n'.join(section_content)
            elif current_section == 'examples':
                examples = '\n'.join(section_content)

        except Exception as e:
            logger.warning(f"Error parsing query components: {e}")

        return schema, knowledge, examples

    def _extract_question_from_query(self, query: str) -> str:
        """从query中提取实际的问题文本"""
        try:
            import re

            # 方法1: 查找"问题："和"，提取中间的内容
            pattern = r'问题：(.+?)，写出对应的SQL语句'
            match = re.search(pattern, query, re.DOTALL)

            if match:
                question = match.group(1).strip()
                logger.debug(f"Extracted question using pattern: {question[:100]}...")
                return question

            # 方法2: 查找"问题："到"答案："之间的内容
            pattern2 = r'问题：(.*?)答案：'
            match2 = re.search(pattern2, query, re.DOTALL)
            if match2:
                question = match2.group(1).strip()
                logger.debug(f"Extracted question using pattern2: {question[:100]}...")
                return question

            # 方法3: 从"问题："之后，到下一个空行或逗号之前
            if '问题：' in query:
                start = query.find('问题：') + 4
                remaining = query[start:]

                # 查找第一个逗号，或者停止在合理长度
                first_comma = remaining.find('，')
                if first_comma > 0 and first_comma < 500:
                    question = remaining[:first_comma].strip()
                    logger.debug(f"Extracted question using comma: {question[:100]}...")
                    return question

                # 如果没有逗号，取前200字符
                if len(remaining) > 0:
                    question = remaining[:200].strip()
                    # 确保不截断到SQL关键字
                    question = question.split('，')[0].strip()
                    logger.debug(f"Extracted question using fixed length: {question[:100]}...")
                    return question

            # 如果以上方法都失败，返回空字符串而不是整个query
            logger.warning(f"Failed to extract question from query, returning empty string")
            return ""

        except Exception as e:
            logger.warning(f"Error extracting question: {e}")
            return ""

    def _parse_training_prompt(self, prompt: str) -> tuple:
        """解析训练提示"""
        try:
            # 这里根据实际构建的提示格式进行解析
            parts = prompt.split('\n\n')
            query = ""
            schema = ""
            knowledge = ""

            for part in parts:
                if part.startswith('问题：'):
                    query = part.replace('问题：', '').strip()
                elif part.startswith('Schema：'):
                    schema = part.replace('Schema：', '').strip()
                elif part.startswith('业务知识：'):
                    knowledge = part.replace('业务知识：', '').strip()

            return query, schema, knowledge, ""
        except Exception as e:
            logger.error(f"Error parsing training prompt: {e}")
            return "", "", "", ""

    def _build_training_prompt(self, query: str, schema: str, knowledge: str, examples: str) -> str:
        """构建训练提示"""
        prompt_parts = [
            f"问题：{query}",
            f"Schema：{schema}",
            f"业务知识：{knowledge}"
        ]
        if examples:
            prompt_parts.append(f"示例：{examples}")
        return '\n\n'.join(prompt_parts)

    def _extract_sql_from_response(self, response: str) -> str:
        """从响应中提取SQL"""
        import re
        sql_match = re.search(r'<sql>(.*?)</sql>', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        return ""

    def _extract_thinking_from_response(self, response: str) -> str:
        """从响应中提取思考过程"""
        import re
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            return thinking_match.group(1).strip()
        return ""

    def _extract_self_eval_from_response(self, response: str) -> str:
        """从响应中提取自评估"""
        import re
        eval_match = re.search(r'<self_eval>(.*?)</self_eval>', response, re.DOTALL | re.IGNORECASE)
        if eval_match:
            return eval_match.group(1).strip()
        return ""


def _process_single_sample(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个样本的函数（用于并行处理）

    Args:
        params: 包含参数的字典

    Returns:
        处理结果字典
    """
    try:
        import sys
        from deepseek_sql.main import DeepSeekNL2SQL

        idx = params['idx']
        query = params['query']
        schema = params['schema']
        knowledge = params['knowledge']
        examples = params['examples']
        reference_type = params['reference_type']
        reference_sql = params['reference_sql']
        model_name = params['model_name']
        pool_dir = params['pool_dir']
        vlm_model_path = params.get('vlm_model_path', None)

        # 初始化DeepSeek系统（每个进程独立，支持VLM）
        deepseek_system = DeepSeekNL2SQL(
            model_name=model_name,
            pool_dir=pool_dir,
            vlm_model_path=vlm_model_path
        )

        # 处理查询
        deepseek_result = deepseek_system.process_query(
            query=query,
            schema=schema,
            knowledge=knowledge,
            examples=examples,
            problem_idx=f"train_{idx}"
        )

        if deepseek_result.get('success', False):
            # 构建训练样本
            prompt_parts = [
                f"问题：{query}",
                f"Schema：{schema}",
                f"业务知识：{knowledge}"
            ]
            if examples:
                prompt_parts.append(f"示例：{examples}")

            training_sample = {
                'prompt': '\n\n'.join(prompt_parts),
                'reference': reference_sql or deepseek_result.get('best_sql', ''),
                'reference_type': reference_type,
                'process_reward_data': deepseek_result.get('process_reward', {}),
                'deepseek_result': deepseek_result
            }

            return {
                'success': True,
                'data': training_sample,
                'idx': idx
            }
        else:
            return {'success': False, 'idx': idx}

    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Process sample {params.get('idx', 'unknown')} failed: {e}")
        return {'success': False, 'idx': params.get('idx', 'unknown'), 'error': str(e)}


def load_config(config_file: str) -> DeepSeekTrainingConfig:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Flatten nested dict if necessary
    if 'training' in config_dict:
        config_dict.update(config_dict['training'])
        del config_dict['training']

    return DeepSeekTrainingConfig(**config_dict)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek GRPO Training")
    parser.add_argument("--config", type=str, default="config_deepseek.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    # Add GPU configuration arguments
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="CUDA visible devices (e.g., 0,1,2,3)")
    parser.add_argument("--gpus_per_node", type=int, default=None, help="Number of GPUs per node")
    parser.add_argument("--tensor_model_parallel_size", type=int, default=None, help="Tensor model parallel size")

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

    # 如果指定了恢复点，更新配置
    if args.resume:
        config.resume_from_checkpoint = args.resume

    # 创建并运行训练器
    trainer = DeepSeekGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()