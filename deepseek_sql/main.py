"""
DeepSeek NL2SQL主控制器

基于DeepSeek-Math-V2的自验证迭代优化机制，实现SQL生成过程的多层验证
这是整个DeepSeek SQL适配模块的核心控制器
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .templates import SQLTemplates
from .generators import SQLGenerator
from .verifiers import SQLVerifier
from .meta_verifiers import SQLMetaVerifier
from .proof_pool import SQLProofPool
from .reward_calculator import SQLProcessRewardCalculator

logger = logging.getLogger(__name__)


class DeepSeekNL2SQL:
    """DeepSeek NL2SQL 主控制器"""

    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", pool_dir: str = "./outputs/sql_proof_pool", vlm_model_path: Optional[str] = None):
        """
        初始化DeepSeek NL2SQL系统

        Args:
            model_name: 基础模型名称
            pool_dir: 证明池存储目录
            vlm_model_path: VLM模型路径（可选，如果提供则使用VLM验证）
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vlm = vlm_model_path is not None

        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        if self.device.type == "cpu":
            self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 初始化各个组件
        self.templates = SQLTemplates()
        self.generator = SQLGenerator(self.model, self.tokenizer)
        self.verifier = SQLVerifier()
        self.meta_verifier = SQLMetaVerifier()
        self.proof_pool = SQLProofPool(pool_dir)
        self.reward_calculator = SQLProcessRewardCalculator()

        # 初始化VLM验证器（如果提供）
        self.vlm_verifier = None
        if self.use_vlm:
            from vlm_verifier import VLMVerifier
            logger.info(f"Initializing VLM verifier from {vlm_model_path}")
            self.vlm_verifier = VLMVerifier(vlm_model_path)
            logger.info("VLM verifier initialized successfully")
        else:
            logger.info("Using model-based verification (no VLM provided)")

        # 迭代配置
        self.max_rounds = 3  # 最大迭代轮次
        self.n_generations_per_round = 2  # 每轮生成数量
        self.n_verifications_per_generation = 2  # 每个生成的验证次数

        logger.info(f"DeepSeek NL2SQL initialized with model: {model_name}, VLM: {self.use_vlm}")

    def process_query(self, query: str, schema: str, knowledge: str,
                     examples: str = "", problem_idx: Optional[str] = None) -> Dict[str, Any]:
        """
        处理NL2SQL查询，执行完整的DeepSeek流程

        Args:
            query: 用户问题
            schema: 数据库Schema
            knowledge: 业务知识
            examples: 示例（可选）
            problem_idx: 问题索引（可选，用于证明池）

        Returns:
            完整的查询处理结果
        """
        try:
            start_time = time.time()

            # 生成问题索引
            if problem_idx is None:
                problem_idx = self._generate_problem_idx(query)

            logger.info(f"Processing query: {query[:100]}...")

            # 训练历史记录
            training_history = []

            # === 第1轮：初始生成 ===
            logger.info("=== Round 1: Initial Generation ===")
            round1_result = self._execute_round(
                round_number=1,
                query=query,
                schema=schema,
                knowledge=knowledge,
                examples=examples,
                problem_idx=problem_idx,
                previous_feedback=None
            )
            training_history.append(round1_result)

            # === 第2轮-N轮：基于反馈的细化 ===
            for round_num in range(2, self.max_rounds + 1):
                logger.info(f"=== Round {round_num}: Refinement ===")

                # 准备细化反馈
                refinement_feedback = self._prepare_refinement_feedback(
                    problem_idx, training_history, round_num
                )

                if not refinement_feedback:
                    logger.info(f"No refinement opportunities found, stopping at round {round_num-1}")
                    break

                # 执行细化轮次
                round_result = self._execute_round(
                    round_number=round_num,
                    query=query,
                    schema=schema,
                    knowledge=knowledge,
                    examples=examples,
                    problem_idx=problem_idx,
                    previous_feedback=refinement_feedback
                )
                training_history.append(round_result)

                # 检查是否达到停止条件
                if self._should_stop_early(training_history):
                    logger.info(f"Early stopping at round {round_num}")
                    break

            # === 最终处理：计算过程奖励和选择最佳结果 ===
            logger.info("=== Final Processing ===")
            process_reward_result = self.reward_calculator.calculate_process_reward(
                problem_idx, query, training_history
            )

            # 获取最佳结果
            best_final_proof = self.proof_pool.get_best_proofs(problem_idx, 1)
            final_result = best_final_proof[0] if best_final_proof else {}

            # 汇总结果
            processing_time = time.time() - start_time

            final_result = {
                'problem_idx': problem_idx,
                'query': query,
                'best_sql': final_result.get('sql', ''),
                'best_thinking': final_result.get('thinking', ''),
                'best_self_eval': final_result.get('self_eval', ''),
                'best_score': final_result.get('final_score', 0.0),
                'process_reward': process_reward_result,
                'training_history': training_history,
                'total_rounds': len(training_history),
                'processing_time': processing_time,
                'success': True
            }

            logger.info(f"Query processing completed in {processing_time:.2f}s")
            return final_result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'problem_idx': problem_idx,
                'query': query,
                'error': str(e),
                'success': False
            }

    def _execute_round(self, round_number: int, query: str, schema: str,
                      knowledge: str, examples: str, problem_idx: str,
                      previous_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行单轮生成-验证流程

        Args:
            round_number: 轮次编号
            query: 用户问题
            schema: 数据库Schema
            knowledge: 业务知识
            examples: 示例
            problem_idx: 问题索引
            previous_feedback: 前几轮的反馈信息

        Returns:
            轮次执行结果
        """
        round_result = {
            'round_number': round_number,
            'generation': None,
            'verifications': [],
            'meta_verifications': [],
            'round_success': False
        }

        try:
            # === 生成阶段 ===
            if round_number == 1:
                # 第一轮：初始生成
                generation_results = []
                for i in range(self.n_generations_per_round):
                    generation = self.generator.generate_sql_with_self_eval(
                        query=query,
                        schema=schema,
                        knowledge=knowledge,
                        examples=examples,
                        temperature=0.7 + (i * 0.1)  # 轻微温度变化
                    )
                    generation_results.append(generation)
            else:
                # 后续轮次：基于反馈的细化
                if previous_feedback:
                    best_previous = previous_feedback.get('best_previous', {})
                    if best_previous:
                        refinement = self.generator.refine_sql(
                            query=query,
                            original_sql=best_previous.get('sql', ''),
                            verification_feedback=previous_feedback.get('feedback_text', ''),
                            average_score=best_previous.get('final_score', 0.0),
                            schema=schema,
                            knowledge=knowledge,
                            temperature=0.6
                        )
                        # 转换细化结果为生成结果格式
                        generation_results = [{
                            'thinking': refinement.get('refined_thinking', ''),
                            'sql': refinement.get('refined_sql', ''),
                            'self_eval': refinement.get('refined_self_eval', ''),
                            'full_response': refinement.get('full_response', ''),
                            'success': refinement.get('success', False),
                            'error': refinement.get('error', '')
                        }]
                    else:
                        generation_results = []
                else:
                    generation_results = []

            if not generation_results:
                logger.warning(f"No generations produced in round {round_number}")
                return round_result

            # 选择最佳生成结果进行验证
            best_generation = max(generation_results, key=lambda x: len(x.get('sql', '')))
            round_result['generation'] = best_generation

            if not best_generation.get('success', True):
                logger.debug(f"Best generation failed in round {round_number}")
                return round_result

            # === 验证阶段 ===
            verification_results = []

            if self.use_vlm and self.vlm_verifier:
                # 使用VLM验证器（独立模型）
                logger.debug("Using VLM verification")
                vlm_verification = self.vlm_verifier.verify_sql(
                    query=query,
                    sql=best_generation.get('sql', ''),
                    schema=schema,
                    knowledge=knowledge
                )
                verification_results.append(vlm_verification)

                # 记录这是VLM验证
                round_result['verification_type'] = 'VLM'
            else:
                # 使用模型-based验证（旧方式）
                logger.debug("Using model-based verification")
                for i in range(self.n_verifications_per_generation):
                    verification = self.verifier.verify_sql(
                        query=query,
                        sql=best_generation.get('sql', ''),
                        schema=schema,
                        knowledge=knowledge,
                        model=self.model,
                        tokenizer=self.tokenizer
                    )
                    verification_results.append(verification)

                round_result['verification_type'] = 'model-based'

            round_result['verifications'] = verification_results

            # === 元验证阶段 ===
            if verification_results:
                meta_verifications = []
                for i in range(min(1, len(verification_results))):  # 对每个验证进行元验证
                    meta_verification = self.meta_verifier.meta_verify(
                        query=query,
                        sql=best_generation.get('sql', ''),
                        verification_results=[verification_results[i]],  # 单验证元验证
                        schema=schema,
                        knowledge=knowledge,
                        model=self.model,
                        tokenizer=self.tokenizer
                    )
                    meta_verifications.append(meta_verification)

                round_result['meta_verifications'] = meta_verifications

            # === 添加到证明池 ===
            proof_data = {
                'sql': best_generation.get('sql', ''),
                'thinking': best_generation.get('thinking', ''),
                'self_eval': best_generation.get('self_eval', ''),
                'verifications': verification_results,
                'meta_verifications': meta_verifications,
                'round_number': round_number,
                'generation_method': 'initial' if round_number == 1 else 'refinement',
                'parent_proof_id': previous_feedback.get('best_previous', {}).get('proof_id') if previous_feedback else None
            }

            proof_id = self.proof_pool.add_proof(problem_idx, proof_data)
            proof_data['proof_id'] = proof_id

            round_result['round_success'] = True
            logger.info(f"Round {round_number} completed successfully")

        except Exception as e:
            logger.error(f"Round {round_number} failed: {e}")
            round_result['error'] = str(e)

        return round_result

    def _prepare_refinement_feedback(self, problem_idx: str, training_history: List[Dict[str, Any]],
                                   current_round: int) -> Optional[Dict[str, Any]]:
        """
        准备细化反馈信息

        Args:
            problem_idx: 问题索引
            training_history: 训练历史
            current_round: 当前轮次

        Returns:
            反馈信息或None（如果没有改进空间）
        """
        try:
            # 获取最佳的前一轮证明
            if current_round <= 2:
                previous_round = training_history[0]  # 第1轮
            else:
                previous_round = training_history[-2]  # 前一轮

            # 获取最佳证明
            best_proofs = self.proof_pool.get_best_proofs(problem_idx, 1)
            if not best_proofs:
                return None

            best_proof = best_proofs[0]
            best_score = best_proof.get('final_score', 0.0)

            # 如果分数已经很高，不需要细化
            if best_score >= 0.95:
                return None

            # 准备反馈文本
            verifications = best_proof.get('verifications', [])
            feedback_parts = []

            for verification in verifications:
                analysis = verification.get('analysis', '')
                issues = verification.get('issues', [])
                score = verification.get('score', 0.0)

                feedback_parts.append(f"评分: {score:.2f}")
                if analysis:
                    feedback_parts.append(f"分析: {analysis}")
                if issues and issues != ['无']:
                    feedback_parts.append(f"问题: {', '.join(issues)}")

            feedback_text = "; ".join(feedback_parts)

            return {
                'best_previous': best_proof,
                'feedback_text': feedback_text,
                'improvement_needed': 1.0 - best_score
            }

        except Exception as e:
            logger.error(f"Failed to prepare refinement feedback: {e}")
            return None

    def _should_stop_early(self, training_history: List[Dict[str, Any]]) -> bool:
        """
        判断是否应该早停

        Args:
            training_history: 训练历史

        Returns:
            是否早停
        """
        if len(training_history) < 2:
            return False

        # 获取最近两轮的最佳分数
        prev_best = self._get_round_best_score(training_history[-2])
        curr_best = self._get_round_best_score(training_history[-1])

        # 如果分数很高，可以早停
        if curr_best >= 0.95:
            return True

        # 如果改进很小，可以早停
        improvement = curr_best - prev_best
        if improvement < 0.05:  # 改进小于5%
            return True

        return False

    def _get_round_best_score(self, round_data: Dict[str, Any]) -> float:
        """获取轮次的最佳分数（与reward_calculator中的方法相同）"""
        meta_verifications = round_data.get('meta_verifications', [])
        if meta_verifications:
            adjusted_scores = [v.get('adjusted_score', v.get('score', 0.0))
                             for v in meta_verifications if v.get('success', True)]
            if adjusted_scores:
                return max(adjusted_scores)

        verifications = round_data.get('verifications', [])
        if verifications:
            verification_scores = [v.get('score', 0.0) for v in verifications if v.get('success', True)]
            if verification_scores:
                return max(verification_scores)

        generation_data = round_data.get('generation', {})
        return generation_data.get('self_eval_score', 0.5)

    def _generate_problem_idx(self, query: str) -> str:
        """生成问题索引"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()[:8]

    def get_pool_statistics(self) -> Dict[str, Any]:
        """获取证明池统计信息"""
        try:
            # 获取所有问题目录
            problem_dirs = [d for d in self.proof_pool.pool_dir.iterdir() if d.is_dir()]

            total_proofs = 0
            problem_indices = []
            best_scores = []

            for problem_dir in problem_dirs:
                problem_idx = problem_dir.name
                problem_indices.append(problem_idx)

                stats = self.proof_pool.get_pool_stats(problem_idx)
                total_proofs += stats.get('pool_size', 0)

                best_proofs = self.proof_pool.get_best_proofs(problem_idx, 1)
                if best_proofs:
                    best_scores.append(best_proofs[0].get('final_score', 0.0))

            return {
                'total_problems': len(problem_indices),
                'total_proofs': total_proofs,
                'avg_proofs_per_problem': total_proofs / len(problem_indices) if problem_indices else 0,
                'avg_best_score': sum(best_scores) / len(best_scores) if best_scores else 0.0,
                'max_best_score': max(best_scores) if best_scores else 0.0,
                'min_best_score': min(best_scores) if best_scores else 0.0
            }

        except Exception as e:
            logger.error(f"Failed to get pool statistics: {e}")
            return {'error': str(e)}

    def clear_all_data(self):
        """清空所有数据"""
        self.proof_pool.clear_pool()
        logger.info("All proof pool data cleared")