"""
元分类器（Meta-Classifier）
验证分类结果的质量
"""

import logging
from typing import Tuple
import re

logger = logging.getLogger(__name__)


class MetaClassifier:
    """
    元分类器

    功能：
    1. 验证复杂度分类是否合理
    2. 评估推理过程质量
    3. 计算置信度分数
    """

    @staticmethod
    def verify_classification(
        original_sql: str,
        predicted_type: str,
        thinking_process: str
    ) -> Tuple[bool, float]:
        """
        验证分类结果是否合理

        Args:
            original_sql: 原始SQL
            predicted_type: 预测的复杂度类型
            thinking_process: 推理过程

        Returns:
            (是否合理, 置信度分数)
        """
        # 检查推理过程质量
        thinking_quality = MetaClassifier._check_thinking_quality(thinking_process)

        # 检查SQL与分类类型的匹配度
        type_match = MetaClassifier._check_type_match(original_sql, predicted_type)

        # 检查SQL的基本有效性
        sql_validity = MetaClassifier._check_sql_validity(original_sql)

        # 综合评分
        confidence = (thinking_quality * 0.35 + type_match * 0.35 + sql_validity * 0.3)

        is_reasonable = confidence > 0.6 and sql_validity > 0.5

        return is_reasonable, confidence

    @staticmethod
    def _check_thinking_quality(thinking: str) -> float:
        """
        检查推理过程的质量

        评估维度：
        1. 长度（至少100字符）
        2. 关键词覆盖
        3. 逻辑连接词
        4. 结构化程度

        Returns:
            质量评分 (0-1)
        """
        if not thinking:
            return 0.0

        quality_score = 0.0

        # 1. 长度检查
        if len(thinking) >= 100:
            quality_score += 0.25
        elif len(thinking) >= 50:
            quality_score += 0.15
        else:
            quality_score += 0.05

        # 2. 关键词检查
        keywords = [
            '约束条件', '表', '字段', '过滤', '聚合',
            'WHERE', 'FROM', 'JOIN', 'GROUP BY', 'SELECT'
        ]
        found_keywords = sum(1 for kw in keywords if kw.lower() in thinking.lower())
        keyword_score = min(found_keywords / len(keywords), 1.0)
        quality_score += keyword_score * 0.35

        # 3. 逻辑连接检查
        logical_connectors = ['因此', '所以', '->', '根据', '基于', '首先', '然后', '最后']
        has_logic = any(conn in thinking for conn in logical_connectors)
        if has_logic:
            quality_score += 0.25
        else:
            quality_score += 0.1

        # 4. 结构化程度
        lines = thinking.split('\n')
        if len(lines) >= 3:  # 至少3行，表示有结构
            quality_score += 0.15
        else:
            quality_score += 0.05

        return min(quality_score, 1.0)

    @staticmethod
    def _check_type_match(sql: str, predicted_type: str) -> float:
        """
        检查SQL与预测类型的匹配度

        计算SQL的复杂度指标：
        - JOIN数量
        - CTE (WITH) 数量
        - UNION数量
        """
        sql_upper = sql.upper()

        join_count = sql_upper.count('JOIN')
        cte_count = sql_upper.count('WITH')
        union_count = sql_upper.count('UNION')

        complexity_score = join_count + cte_count * 2 + union_count

        if predicted_type == "多步推理":
            if complexity_score >= 2:
                return 1.0
            elif complexity_score == 1:
                return 0.7
            else:
                return 0.3
        else:  # "sql"
            if complexity_score <= 1:
                return 1.0
            elif complexity_score == 2:
                return 0.6
            else:
                return 0.2

    @staticmethod
    def _check_sql_validity(sql: str) -> float:
        """
        检查SQL的基本有效性

        检查项：
        1. SELECT 关键字
        2. FROM 关键字
        3. 括号匹配
        4. 关键字顺序

        Returns:
            有效性评分 (0-1)
        """
        sql_upper = sql.upper()
        validity_score = 0.0

        checks = {
            'select': 'SELECT' in sql_upper,
            'from': 'FROM' in sql_upper,
            'brackets_balanced': sql.count('(') == sql.count(')'),
            'correct_order': sql_upper.find('SELECT') < sql_upper.find('FROM'),
        }

        for check, is_valid in checks.items():
            if is_valid:
                validity_score += 0.25

        # 检查是否有明显的语法错误
        error_patterns = [
            r'SELECT\s+FROM',  # SELECT后直接FROM
            r'FROM\s+FROM',    # 两个FROM
            r'WHERE\s+WHERE',  # 两个WHERE
        ]

        for pattern in error_patterns:
            if re.search(pattern, sql_upper):
                validity_score -= 0.1

        return max(0.0, min(validity_score, 1.0))

    @classmethod
    def get_meta_verification_report(cls,
                                     original_sql: str,
                                     predicted_type: str,
                                     thinking_process: str) -> dict:
        """
        获取元验证报告

        Returns:
            详细的验证报告
        """
        is_reasonable, confidence = cls.verify_classification(
            original_sql, predicted_type, thinking_process
        )

        thinking_quality = cls._check_thinking_quality(thinking_process)
        type_match = cls._check_type_match(original_sql, predicted_type)
        sql_validity = cls._check_sql_validity(original_sql)

        return {
            'is_reasonable': is_reasonable,
            'overall_confidence': confidence,
            'thinking_quality': thinking_quality,
            'type_match': type_match,
            'sql_validity': sql_validity,
            'status': '✅ 合理' if is_reasonable else '⚠️ 需要审查',
            'details': {
                'thinking_quality': f"{thinking_quality * 100:.1f}%",
                'type_match': f"{type_match * 100:.1f}%",
                'sql_validity': f"{sql_validity * 100:.1f}%",
            }
        }
