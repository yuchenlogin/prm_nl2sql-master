"""
过程奖励计算器

基于DeepSeek-Math-V2的过程奖励思想，计算SQL生成过程中的多维度奖励
包括：生成奖励、验证奖励、元验证奖励、迭代奖励
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .proof_pool import SQLProofPool

logger = logging.getLogger(__name__)


class SQLProcessRewardCalculator:
    """SQL过程奖励计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化过程奖励计算器

        Args:
            config: 奖励配置
        """
        self.config = config or self._default_config()
        self.proof_pool = SQLProofPool()

        # 奖励权重
        self.reward_weights = self.config.get('reward_weights', {
            'generation': 0.25,      # 生成阶段奖励
            'verification': 0.30,    # 验证阶段奖励
            'meta_verification': 0.25, # 元验证阶段奖励
            'iteration': 0.20        # 迭代改进奖励
        })

        # 过程奖励参数
        self.process_params = self.config.get('process_params', {
            'score_decay_factor': 0.9,    # 分数衰减因子
            'improvement_bonus': 0.2,     # 改进奖励
            'consistency_bonus': 0.15,    # 一致性奖励
            'early_termination_penalty': 0.3  # 早期终止惩罚
        })

    def calculate_process_reward(self, problem_idx: str, query: str,
                                training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算整个过程的活动奖励

        Args:
            problem_idx: 问题索引
            query: 用户问题
            training_history: 训练历史，包含多轮生成和验证结果

        Returns:
            过程奖励结果字典
        """
        try:
            # 1. 计算各阶段奖励
            generation_rewards = self._calculate_generation_rewards(training_history)
            verification_rewards = self._calculate_verification_rewards(training_history)
            meta_verification_rewards = self._calculate_meta_verification_rewards(training_history)
            iteration_rewards = self._calculate_iteration_rewards(training_history)

            # 2. 计算总过程奖励
            total_process_reward = (
                generation_rewards['total'] * self.reward_weights['generation'] +
                verification_rewards['total'] * self.reward_weights['verification'] +
                meta_verification_rewards['total'] * self.reward_weights['meta_verification'] +
                iteration_rewards['total'] * self.reward_weights['iteration']
            )

            # 3. 分解奖励组成
            reward_breakdown = {
                'generation': {
                    'score': generation_rewards['total'],
                    'weight': self.reward_weights['generation'],
                    'contribution': generation_rewards['total'] * self.reward_weights['generation'],
                    'details': generation_rewards['details']
                },
                'verification': {
                    'score': verification_rewards['total'],
                    'weight': self.reward_weights['verification'],
                    'contribution': verification_rewards['total'] * self.reward_weights['verification'],
                    'details': verification_rewards['details']
                },
                'meta_verification': {
                    'score': meta_verification_rewards['total'],
                    'weight': self.reward_weights['meta_verification'],
                    'contribution': meta_verification_rewards['total'] * self.reward_weights['meta_verification'],
                    'details': meta_verification_rewards['details']
                },
                'iteration': {
                    'score': iteration_rewards['total'],
                    'weight': self.reward_weights['iteration'],
                    'contribution': iteration_rewards['total'] * self.reward_weights['iteration'],
                    'details': iteration_rewards['details']
                }
            }

            # 4. 获取最佳最终结果
            best_final_result = self._get_best_final_result(problem_idx)

            return {
                'total_process_reward': total_process_reward,
                'reward_breakdown': reward_breakdown,
                'best_final_sql': best_final_result.get('sql', ''),
                'best_final_score': best_final_result.get('final_score', 0.0),
                'total_rounds': len(training_history),
                'process_quality': self._assess_process_quality(training_history),
                'success': True
            }

        except Exception as e:
            logger.error(f"Failed to calculate process reward: {e}")
            return {
                'total_process_reward': 0.0,
                'error': str(e),
                'success': False
            }

    def _calculate_generation_rewards(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算生成阶段奖励"""
        generation_rewards = []
        total_reward = 0.0

        for round_data in training_history:
            generation_data = round_data.get('generation', {})
            if not generation_data:
                continue

            # 生成质量奖励
            generation_quality = self._assess_generation_quality(generation_data)

            # 自评估准确性奖励
            self_eval_accuracy = self._assess_self_eval_accuracy(generation_data, round_data)

            # 轮次生成奖励
            round_generation_reward = (generation_quality * 0.7 + self_eval_accuracy * 0.3)

            # 应用衰减因子（后期生成的奖励较小）
            round_idx = round_data.get('round_number', 0)
            decayed_reward = round_generation_reward * (self.process_params['score_decay_factor'] ** round_idx)

            generation_rewards.append({
                'round': round_idx,
                'quality': generation_quality,
                'self_eval_accuracy': self_eval_accuracy,
                'round_reward': round_generation_reward,
                'decayed_reward': decayed_reward
            })

            total_reward += decayed_reward

        return {
            'total': total_reward,
            'per_round': generation_rewards,
            'details': {
                'avg_quality': sum(r['quality'] for r in generation_rewards) / len(generation_rewards) if generation_rewards else 0.0,
                'avg_self_eval_accuracy': sum(r['self_eval_accuracy'] for r in generation_rewards) / len(generation_rewards) if generation_rewards else 0.0
            }
        }

    def _calculate_verification_rewards(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算验证阶段奖励"""
        verification_rewards = []
        total_reward = 0.0

        for round_data in training_history:
            verifications = round_data.get('verifications', [])
            if not verifications:
                continue

            # 验证质量奖励
            verification_quality = self._assess_verification_quality(verifications)

            # 验证一致性奖励
            consistency_reward = self._assess_verification_consistency(verifications)

            # 轮次验证奖励
            round_verification_reward = (verification_quality * 0.6 + consistency_reward * 0.4)

            round_idx = round_data.get('round_number', 0)
            decayed_reward = round_verification_reward * (self.process_params['score_decay_factor'] ** round_idx)

            verification_rewards.append({
                'round': round_idx,
                'quality': verification_quality,
                'consistency': consistency_reward,
                'round_reward': round_verification_reward,
                'decayed_reward': decayed_reward,
                'verification_count': len(verifications)
            })

            total_reward += decayed_reward

        return {
            'total': total_reward,
            'per_round': verification_rewards,
            'details': {
                'avg_quality': sum(r['quality'] for r in verification_rewards) / len(verification_rewards) if verification_rewards else 0.0,
                'avg_consistency': sum(r['consistency'] for r in verification_rewards) / len(verification_rewards) if verification_rewards else 0.0
            }
        }

    def _calculate_meta_verification_rewards(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算元验证阶段奖励"""
        meta_verification_rewards = []
        total_reward = 0.0

        for round_data in training_history:
            meta_verifications = round_data.get('meta_verifications', [])
            if not meta_verifications:
                continue

            # 元验证质量奖励
            meta_quality = self._assess_meta_verification_quality(meta_verifications)

            # 置信度奖励
            confidence_reward = self._assess_confidence_quality(meta_verifications)

            # 轮次元验证奖励
            round_meta_reward = (meta_quality * 0.7 + confidence_reward * 0.3)

            round_idx = round_data.get('round_number', 0)
            decayed_reward = round_meta_reward * (self.process_params['score_decay_factor'] ** round_idx)

            meta_verification_rewards.append({
                'round': round_idx,
                'quality': meta_quality,
                'confidence': confidence_reward,
                'round_reward': round_meta_reward,
                'decayed_reward': decayed_reward
            })

            total_reward += decayed_reward

        return {
            'total': total_reward,
            'per_round': meta_verification_rewards,
            'details': {
                'avg_quality': sum(r['quality'] for r in meta_verification_rewards) / len(meta_verification_rewards) if meta_verification_rewards else 0.0,
                'avg_confidence': sum(r['confidence'] for r in meta_verification_rewards) / len(meta_verification_rewards) if meta_verification_rewards else 0.0
            }
        }

    def _calculate_iteration_rewards(self, training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算迭代改进奖励"""
        iteration_rewards = []
        total_reward = 0.0

        # 计算连续轮次间的改进
        for i in range(1, len(training_history)):
            prev_round = training_history[i-1]
            curr_round = training_history[i]

            # 分数改进奖励
            score_improvement = self._calculate_score_improvement(prev_round, curr_round)

            # 逻辑改进奖励
            logic_improvement = self._calculate_logic_improvement(prev_round, curr_round)

            # 轮次迭代奖励
            round_iteration_reward = (score_improvement * 0.6 + logic_improvement * 0.4)

            # 应用改进奖励
            improved_reward = round_iteration_reward * (1 + self.process_params['improvement_bonus'])

            round_idx = curr_round.get('round_number', i)
            decayed_reward = improved_reward * (self.process_params['score_decay_factor'] ** round_idx)

            iteration_rewards.append({
                'round': round_idx,
                'score_improvement': score_improvement,
                'logic_improvement': logic_improvement,
                'round_reward': round_iteration_reward,
                'improved_reward': improved_reward,
                'decayed_reward': decayed_reward
            })

            total_reward += decayed_reward

        # 整体迭代质量奖励
        overall_iteration_quality = self._assess_overall_iteration_quality(training_history)
        total_reward += overall_iteration_quality * self.process_params['consistency_bonus']

        return {
            'total': total_reward,
            'per_round': iteration_rewards,
            'details': {
                'avg_score_improvement': sum(r['score_improvement'] for r in iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0,
                'avg_logic_improvement': sum(r['logic_improvement'] for r in iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0
            }
        }

    def _assess_generation_quality(self, generation_data: Dict[str, Any]) -> float:
        """评估生成质量"""
        quality_factors = []

        # SQL完整性
        sql = generation_data.get('sql', '')
        if sql and sql.strip():
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.0)

        # 推理质量
        thinking = generation_data.get('thinking', '')
        if thinking and len(thinking.strip()) > 50:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.3)

        # 自评估完整性
        self_eval = generation_data.get('self_eval', '')
        if self_eval and len(self_eval.strip()) > 20:
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.2)

        return sum(quality_factors) / len(quality_factors)

    def _assess_self_eval_accuracy(self, generation_data: Dict[str, Any], round_data: Dict[str, Any]) -> float:
        """评估自评估准确性"""
        # 获取自评估分数
        self_eval_score = generation_data.get('self_eval_score', 0.5)

        # 获取验证的客观分数
        verifications = round_data.get('verifications', [])
        if verifications:
            objective_scores = [v.get('score', 0.0) for v in verifications if v.get('success', True)]
            if objective_scores:
                objective_score = sum(objective_scores) / len(objective_scores)

                # 计算自评估与客观评估的差距
                accuracy = 1.0 - abs(self_eval_score - objective_score)
                return max(0.0, accuracy)

        return 0.5  # 默认值

    def _assess_verification_quality(self, verifications: List[Dict[str, Any]]) -> float:
        """评估验证质量"""
        if not verifications:
            return 0.0

        successful_verifications = [v for v in verifications if v.get('success', True)]
        if not successful_verifications:
            return 0.0

        # 基于验证成功率
        success_rate = len(successful_verifications) / len(verifications)

        # 基于验证分析的详细程度
        avg_analysis_length = sum(len(v.get('analysis', '')) for v in successful_verifications) / len(successful_verifications)
        analysis_quality = min(avg_analysis_length / 100, 1.0)  # 假设100字符为满分

        return (success_rate * 0.6 + analysis_quality * 0.4)

    def _assess_verification_consistency(self, verifications: List[Dict[str, Any]]) -> float:
        """评估验证一致性"""
        if len(verifications) < 2:
            return 1.0  # 单个验证默认一致

        scores = [v.get('score', 0.0) for v in verifications if v.get('success', True)]
        if len(scores) < 2:
            return 1.0

        # 计算分数标准差
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # 标准差越小，一致性越高
        consistency = max(0.0, 1.0 - std_dev * 2)
        return consistency

    def _assess_meta_verification_quality(self, meta_verifications: List[Dict[str, Any]]) -> float:
        """评估元验证质量"""
        if not meta_verifications:
            return 0.5  # 默认值

        successful_meta = [v for v in meta_verifications if v.get('success', True)]
        if not successful_meta:
            return 0.0

        # 基于置信度
        confidence_scores = []
        confidence_map = {'high': 1.0, 'medium': 0.5, 'low': 0.2}
        for meta in successful_meta:
            confidence = meta.get('confidence', 'medium')
            confidence_scores.append(confidence_map.get(confidence, 0.5))

        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        # 基于成功率
        success_rate = len(successful_meta) / len(meta_verifications)

        return (avg_confidence * 0.7 + success_rate * 0.3)

    def _assess_confidence_quality(self, meta_verifications: List[Dict[str, Any]]) -> float:
        """评估置信度质量"""
        if not meta_verifications:
            return 0.5

        high_confidence_count = sum(1 for v in meta_verifications if v.get('confidence') == 'high')
        total_count = len(meta_verifications)

        return high_confidence_count / total_count if total_count > 0 else 0.5

    def _calculate_score_improvement(self, prev_round: Dict[str, Any], curr_round: Dict[str, Any]) -> float:
        """计算分数改进"""
        # 获取前一轮的最佳分数
        prev_best_score = self._get_round_best_score(prev_round)

        # 获取当前轮的最佳分数
        curr_best_score = self._get_round_best_score(curr_round)

        # 计算改进
        improvement = curr_best_score - prev_best_score

        # 正向改进给予奖励，负向改进给予惩罚
        if improvement > 0:
            return min(improvement * 2, 1.0)  # 改进奖励
        else:
            return max(improvement, -0.5)  # 改进惩罚，但不低于-0.5

    def _calculate_logic_improvement(self, prev_round: Dict[str, Any], curr_round: Dict[str, Any]) -> float:
        """计算逻辑改进"""
        # 基于问题减少的数量
        prev_issues = self._count_round_issues(prev_round)
        curr_issues = self._count_round_issues(curr_round)

        if prev_issues == 0 and curr_issues == 0:
            return 0.8  # 两者都无问题，保持高质量

        if prev_issues > 0:
            issue_reduction = (prev_issues - curr_issues) / prev_issues
            return max(0.0, issue_reduction)
        else:
            # 前一轮没有问题，当前轮有问题，给予惩罚
            return -0.3

    def _assess_overall_iteration_quality(self, training_history: List[Dict[str, Any]]) -> float:
        """评估整体迭代质量"""
        if len(training_history) < 2:
            return 0.5

        # 计算整体的改进趋势
        first_score = self._get_round_best_score(training_history[0])
        last_score = self._get_round_best_score(training_history[-1])

        overall_improvement = last_score - first_score

        # 检查是否早期终止（即没有达到应有的迭代次数）
        expected_rounds = min(3, len(training_history))  # 期望至少3轮
        actual_rounds = len(training_history)

        if actual_rounds < expected_rounds:
            # 早期终止惩罚
            early_termination_penalty = self.process_params['early_termination_penalty']
            overall_improvement -= early_termination_penalty

        return max(0.0, min(1.0, overall_improvement))

    def _get_round_best_score(self, round_data: Dict[str, Any]) -> float:
        """获取轮次的最佳分数"""
        # 优先使用元验证的调整分数
        meta_verifications = round_data.get('meta_verifications', [])
        if meta_verifications:
            adjusted_scores = [v.get('adjusted_score', v.get('score', 0.0))
                             for v in meta_verifications if v.get('success', True)]
            if adjusted_scores:
                return max(adjusted_scores)

        # 其次使用验证分数
        verifications = round_data.get('verifications', [])
        if verifications:
            verification_scores = [v.get('score', 0.0) for v in verifications if v.get('success', True)]
            if verification_scores:
                return max(verification_scores)

        # 最后使用生成分数
        generation_data = round_data.get('generation', {})
        return generation_data.get('self_eval_score', 0.5)

    def _count_round_issues(self, round_data: Dict[str, Any]) -> int:
        """计算轮次的问题数量"""
        verifications = round_data.get('verifications', [])
        total_issues = 0

        for verification in verifications:
            issues = verification.get('issues', [])
            if issues and 'none' not in str(issues).lower():
                total_issues += len(issues)

        return total_issues

    def _get_best_final_result(self, problem_idx: str) -> Dict[str, Any]:
        """从证明池获取最佳最终结果"""
        try:
            best_proofs = self.proof_pool.get_best_proofs(problem_idx, 1)
            return best_proofs[0] if best_proofs else {}
        except Exception as e:
            logger.error(f"Failed to get best final result: {e}")
            return {}

    def _assess_process_quality(self, training_history: List[Dict[str, Any]]) -> str:
        """评估整体过程质量"""
        # 计算最终最佳分数
        final_scores = [self._get_round_best_score(round_data) for round_data in training_history]
        if not final_scores:
            return "unknown"

        best_final_score = max(final_scores)

        # 根据分数评估质量
        if best_final_score >= 0.9:
            return "excellent"
        elif best_final_score >= 0.7:
            return "good"
        elif best_final_score >= 0.5:
            return "fair"
        else:
            return "poor"

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'reward_weights': {
                'generation': 0.25,
                'verification': 0.30,
                'meta_verification': 0.25,
                'iteration': 0.20
            },
            'process_params': {
                'score_decay_factor': 0.9,
                'improvement_bonus': 0.2,
                'consistency_bonus': 0.15,
                'early_termination_penalty': 0.3
            }
        }