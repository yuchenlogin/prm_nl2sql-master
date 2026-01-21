"""
过程奖励模型 (Process Reward Model)
评估SQL生成过程中的多个维度，支持8种任务类型
"""

import logging
from typing import Dict, Set
from classifiers.complexity_classifier import TaskTypeClassifier
from classifiers.meta_classifier import MetaClassifier

logger = logging.getLogger(__name__)


class ProcessRewardModel:
    """
    过程奖励模型

    功能：
    1. 计算类型准确度奖励 (type_reward) - 支持8种任务类型
    2. 计算推理过程质量奖励 (thinking_reward)
    3. 计算自我评估准确度奖励 (self_assessment_reward)
    4. 计算SQL结构质量奖励 (sql_structure_reward)

    总奖励 = 0.20 * type_reward +
             0.25 * thinking_reward +
             0.25 * self_assessment_reward +
             0.30 * sql_structure_reward
    """

    # 支持的8种任务类型
    SUPPORTED_TASK_TYPES = {
        "SQL",  # 简单思考直接输出SQL
        "多步推理",  # 多步骤思考，输出可能带有CTE等的复杂SQL
        "反思",  # 将输入的错误SQL更正
        "歧义澄清",  # 用户问题包含歧义点，触发模型思考
        "维度拒识",  # 用户问题包含查询不支持的维度时模型拒绝回答
        "维度退化",  # 维表退化到事实表时仍支持查询
        "指标拒识",  # 用户问题包含查询不支持的指标时模型拒绝回答
        "追问"  # 用户问题不满足查询的必备要求
    }

    # 可训练的任务类型（非拒识类）
    TRAINABLE_TASK_TYPES = {
        "SQL",
        "多步推理",
        "反思",
        "维度退化"
    }

    # 非训练任务类型（拒识类）
    NON_TRAINABLE_TASK_TYPES = {
        "歧义澄清",
        "维度拒识",
        "指标拒识",
        "追问"
    }

    def __init__(self,
                 type_weight: float = 0.20,
                 thinking_weight: float = 0.25,
                 self_assessment_weight: float = 0.25,
                 sql_structure_weight: float = 0.30):
        """
        初始化奖励模型

        Args:
            type_weight: 类型奖励权重
            thinking_weight: 推理过程奖励权重
            self_assessment_weight: 自我评估奖励权重
            sql_structure_weight: SQL结构奖励权重
        """
        self.type_weight = type_weight
        self.thinking_weight = thinking_weight
        self.self_assessment_weight = self_assessment_weight
        self.sql_structure_weight = sql_structure_weight

        # 确保权重和为1
        total_weight = type_weight + thinking_weight + self_assessment_weight + sql_structure_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"权重总和不为1.0: {total_weight}, 正在规范化")
            self.type_weight /= total_weight
            self.thinking_weight /= total_weight
            self.self_assessment_weight /= total_weight
            self.sql_structure_weight /= total_weight

        self.task_type_classifier = TaskTypeClassifier()
        self.meta_classifier = MetaClassifier()

    def compute_reward(self,
                       generated_sql: str,
                       predicted_type: str,
                       thinking: str,
                       reference_type: str,
                       reference_sql: str = None,
                       query: str = None) -> Dict:
        """
        计算总体奖励

        Args:
            generated_sql: 生成的SQL
            predicted_type: 预测的任务类型
            thinking: 推理过程
            reference_type: 参考类型（真实标签）
            reference_sql: 参考SQL（可选）
            query: 用户问题（可选，用于任务类型判断）

        Returns:
            包含各部分奖励的字典
        """
        # 验证任务类型是否有效
        predicted_type = self._validate_task_type(predicted_type)
        reference_type = self._validate_task_type(reference_type)

        # 计算各部分奖励
        type_reward = self._compute_type_reward(predicted_type, reference_type)
        thinking_reward = self._compute_thinking_reward(thinking, predicted_type)
        self_assessment_reward = self._compute_self_assessment_reward(
            generated_sql, thinking, query, predicted_type
        )
        sql_structure_reward = self._compute_sql_structure_reward(
            generated_sql, predicted_type
        )

        # 计算总奖励
        total_reward = (
            self.type_weight * type_reward +
            self.thinking_weight * thinking_reward +
            self.self_assessment_weight * self_assessment_reward +
            self.sql_structure_weight * sql_structure_reward
        )

        result = {
            'total_reward': total_reward,
            'type_reward': type_reward,
            'thinking_reward': thinking_reward,
            'self_assessment_reward': self_assessment_reward,
            'sql_structure_reward': sql_structure_reward,
            'predicted_type': predicted_type,
            'reference_type': reference_type,
            'is_trainable': reference_type in self.TRAINABLE_TASK_TYPES,
            'reward_breakdown': {
                'type': f"{type_reward:.4f} × {self.type_weight:.2f}",
                'thinking': f"{thinking_reward:.4f} × {self.thinking_weight:.2f}",
                'self_assessment': f"{self_assessment_reward:.4f} × {self.self_assessment_weight:.2f}",
                'sql_structure': f"{sql_structure_reward:.4f} × {self.sql_structure_weight:.2f}",
            }
        }

        return result

    def _validate_task_type(self, task_type: str) -> str:
        """
        验证任务类型是否有效

        Args:
            task_type: 任务类型

        Returns:
            有效的任务类型，如果无效则返回"SQL"
        """
        if task_type not in self.SUPPORTED_TASK_TYPES:
            logger.warning(f"未知的任务类型 '{task_type}'，使用默认值 'SQL'")
            return "SQL"
        return task_type

    def _compute_type_reward(self, predicted_type: str, reference_type: str) -> float:
        """
        计算类型预测奖励

        完全匹配 → 1.0
        同类别匹配（如都是可训练类型或都是拒识类型） → 0.5
        完全不匹配 → 0.0

        Args:
            predicted_type: 预测的类型
            reference_type: 参考类型

        Returns:
            类型奖励 (0-1)
        """
        if predicted_type == reference_type:
            return 1.0

        # 检查是否属于同一大类（可训练 vs 非训练）
        predicted_trainable = predicted_type in self.TRAINABLE_TASK_TYPES
        reference_trainable = reference_type in self.TRAINABLE_TASK_TYPES

        if predicted_trainable == reference_trainable:
            # 同属于可训练类型或同属于拒识类型
            return 0.5
        else:
            # 一个可训练，一个拒识
            return 0.0

    def _compute_thinking_reward(self, thinking: str, task_type: str) -> float:
        """
        计算推理过程质量奖励

        评估维度：
        1. 长度（至少100字符）
        2. 关键词覆盖
        3. 逻辑连接词
        4. 结构化程度
        5. 任务类型特定的推理质量

        Args:
            thinking: 推理过程文本
            task_type: 任务类型（用于调整评分标准）

        Returns:
            推理奖励 (0-1)
        """
        if not thinking:
            return 0.0

        # 使用meta_classifier中的思想质量检查
        base_quality = self.meta_classifier._check_thinking_quality(thinking)

        # 根据任务类型调整奖励
        if task_type in self.NON_TRAINABLE_TASK_TYPES:
            # 对于拒识类任务，检查是否包含合理的拒识推理
            has_reasoning = any(
                keyword in thinking
                for keyword in ['不支持', '不存在', '歧义', '缺少', '需要明确', '无法']
            )
            if has_reasoning:
                base_quality = min(base_quality + 0.2, 1.0)

        return base_quality

    def _compute_self_assessment_reward(self, sql: str, thinking: str, query: str = None, task_type: str = None) -> float:
        """
        计算自我评估奖励

        评估模型是否能够正确识别SQL中的问题或任务特性

        Args:
            sql: SQL查询
            thinking: 推理过程
            query: 用户问题（可选）
            task_type: 任务类型（可选）

        Returns:
            自我评估奖励 (0-1)
        """
        # 对于拒识类任务，特殊处理
        if task_type in self.NON_TRAINABLE_TASK_TYPES:
            # 检查推理中是否包含合理的拒识逻辑
            has_refusal_reasoning = any(
                keyword in thinking
                for keyword in ['不支持', '不存在', '歧义', '缺少', '需要明确', '无法']
            )

            # 对于某些任务类型，不需要SQL或SQL为空是正常的
            if not sql and has_refusal_reasoning:
                return 0.9
            elif sql and has_refusal_reasoning:
                # 可能是提供了替代答案
                return 0.7
            else:
                return 0.3

        # 运行任务类型分类器得到问题列表
        classification_result = self.task_type_classifier.classify(sql, query, task_type)

        # 检查推理中是否提及了重要问题
        actual_issues = [
            issue.issue_type for issue in classification_result.issue_details
        ]

        if not actual_issues:
            # 没有问题，评估应该反映这一点
            has_positive_assessment = any(
                word in thinking.lower()
                for word in ['正确', '良好', '完善', '完整', '合适']
            )
            if has_positive_assessment:
                return 0.9
            else:
                return 0.5
        else:
            # 有问题，检查推理中是否提及了它们
            mentioned_issues = 0
            for issue_type in actual_issues:
                # 转换问题类型为可读描述
                issue_desc = self._issue_type_to_description(issue_type)
                if issue_desc.lower() in thinking.lower():
                    mentioned_issues += 1

            # 如果识别出至少50%的问题
            if actual_issues:
                recall = mentioned_issues / len(actual_issues)
                return min(recall + 0.3, 1.0)  # 基础分0.3 + 识别率
            else:
                return 0.5

    def _compute_sql_structure_reward(self, sql: str, task_type: str = None) -> float:
        """
        计算SQL结构质量奖励

        评估SQL的：
        1. 基本有效性 (SELECT, FROM, 括号匹配)
        2. 问题严重程度
        3. 语法正确性
        4. 任务类型特定的标准

        Args:
            sql: SQL查询
            task_type: 任务类型（可选）

        Returns:
            SQL结构奖励 (0-1)
        """
        # 对于拒识类任务，SQL可能为空或非标准
        if task_type in self.NON_TRAINABLE_TASK_TYPES:
            if not sql:
                # 拒识类任务不需要SQL，给予满分
                return 1.0
            else:
                # 如果提供了SQL，可能是替代答案，给予较低但合理的分数
                return 0.7

        # 运行任务类型分类器
        classification_result = self.task_type_classifier.classify(sql)

        # 使用SQL有效性检查
        sql_validity = self.meta_classifier._check_sql_validity(sql)

        # 基于问题严重程度进行惩罚
        severity_penalty = classification_result.severity_score

        # 综合计算
        structure_reward = sql_validity * (1 - severity_penalty * 0.5)

        return max(0.0, min(structure_reward, 1.0))

    @staticmethod
    def _issue_type_to_description(issue_type: str) -> str:
        """
        将问题类型转换为可读描述

        Args:
            issue_type: 问题类型

        Returns:
            问题描述
        """
        descriptions = {
            'missing_where': '缺少WHERE条件',
            'missing_time_range': '缺少时间范围',
            'incorrect_join': 'JOIN缺少ON条件',
            'missing_join_condition': 'JOIN条件不完整',
            'inconsistent_alias': '表别名不一致',
            'missing_group_by': '缺少GROUP BY',
            'unclosed_parenthesis': '括号不匹配',
            'empty_in_clause': 'IN子句为空',
            'invalid_date_format': '日期格式不正确',
            'multiple_tables_no_join': '多个表但未使用JOIN',
            'order_by_without_limit': 'ORDER BY但无LIMIT',
        }
        return descriptions.get(issue_type, issue_type)

    def compute_batch_rewards(self,
                              samples: list) -> list:
        """
        批量计算奖励

        Args:
            samples: 样本列表，每个样本是字典：
                {
                    'generated_sql': str,
                    'predicted_type': str,
                    'thinking': str,
                    'reference_type': str,
                    'reference_sql': str (可选)
                }

        Returns:
            奖励列表
        """
        rewards = []
        for sample in samples:
            reward = self.compute_reward(
                generated_sql=sample['generated_sql'],
                predicted_type=sample['predicted_type'],
                thinking=sample['thinking'],
                reference_type=sample['reference_type'],
                reference_sql=sample.get('reference_sql', '')
            )
            rewards.append(reward)
        return rewards

    def get_reward_stats(self, reward_list: list) -> Dict:
        """
        获取奖励统计信息

        Args:
            reward_list: 奖励列表

        Returns:
            统计字典
        """
        if not reward_list:
            return {}

        total_rewards = [r['total_reward'] for r in reward_list]
        type_rewards = [r['type_reward'] for r in reward_list]
        thinking_rewards = [r['thinking_reward'] for r in reward_list]
        sa_rewards = [r['self_assessment_reward'] for r in reward_list]
        sql_rewards = [r['sql_structure_reward'] for r in reward_list]

        return {
            'total_reward': {
                'mean': sum(total_rewards) / len(total_rewards),
                'min': min(total_rewards),
                'max': max(total_rewards),
                'std': self._std_dev(total_rewards)
            },
            'type_reward': {
                'mean': sum(type_rewards) / len(type_rewards),
                'min': min(type_rewards),
                'max': max(type_rewards),
            },
            'thinking_reward': {
                'mean': sum(thinking_rewards) / len(thinking_rewards),
                'min': min(thinking_rewards),
                'max': max(thinking_rewards),
            },
            'self_assessment_reward': {
                'mean': sum(sa_rewards) / len(sa_rewards),
                'min': min(sa_rewards),
                'max': max(sa_rewards),
            },
            'sql_structure_reward': {
                'mean': sum(sql_rewards) / len(sql_rewards),
                'min': min(sql_rewards),
                'max': max(sql_rewards),
            }
        }

    @staticmethod
    def _std_dev(values: list) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
