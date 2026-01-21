"""
评估指标计算模块
计算各种评估指标
"""

import logging
from typing import Dict, List, Tuple
import re
from classifiers.complexity_classifier import ComplexityClassifier
from classifiers.meta_classifier import MetaClassifier

logger = logging.getLogger(__name__)


class Metrics:
    """评估指标计算类"""

    @staticmethod
    def type_accuracy(predicted_types: List[str], reference_types: List[str]) -> float:
        """
        计算类型分类准确率

        Args:
            predicted_types: 预测的类型列表
            reference_types: 参考类型列表

        Returns:
            准确率 (0-1)
        """
        if not predicted_types or len(predicted_types) != len(reference_types):
            return 0.0

        correct = sum(
            1 for pred, ref in zip(predicted_types, reference_types)
            if pred == ref
        )
        return correct / len(predicted_types)

    @staticmethod
    def sql_validity_score(sql_list: List[str]) -> float:
        """
        计算SQL有效性评分

        评估标准：
        - SELECT 和 FROM 存在
        - 括号匹配
        - 没有明显语法错误

        Args:
            sql_list: SQL查询列表

        Returns:
            平均有效性评分 (0-1)
        """
        classifier = ComplexityClassifier()
        validity_scores = []

        for sql in sql_list:
            classification = classifier.classify(sql)
            # 使用元分类器检查SQL有效性
            meta_classifier = MetaClassifier()
            validity = meta_classifier._check_sql_validity(sql)
            validity_scores.append(validity)

        return sum(validity_scores) / len(validity_scores) if validity_scores else 0.0

    @staticmethod
    def thinking_quality_score(thinking_list: List[str]) -> float:
        """
        计算推理过程质量评分

        Args:
            thinking_list: 推理过程文本列表

        Returns:
            平均质量评分 (0-1)
        """
        meta_classifier = MetaClassifier()
        quality_scores = []

        for thinking in thinking_list:
            quality = meta_classifier._check_thinking_quality(thinking)
            quality_scores.append(quality)

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    @staticmethod
    def self_assessment_accuracy(thinking_list: List[str],
                                 sql_list: List[str]) -> float:
        """
        计算自我评估准确率

        检查推理中是否正确识别了SQL中的问题

        Args:
            thinking_list: 推理过程列表
            sql_list: SQL查询列表

        Returns:
            准确率 (0-1)
        """
        if not thinking_list or len(thinking_list) != len(sql_list):
            return 0.0

        classifier = ComplexityClassifier()
        accuracies = []

        for thinking, sql in zip(thinking_list, sql_list):
            classification = classifier.classify(sql)
            actual_issues = len(classification.issue_details)

            # 检查推理中是否提及了问题
            issue_keywords = [
                '问题', '错误', '缺少', '不匹配', '不一致',
                'WHERE', 'JOIN', 'GROUP BY', '括号', '日期'
            ]
            mentioned_issues = sum(
                1 for keyword in issue_keywords
                if keyword.lower() in thinking.lower()
            )

            if actual_issues == 0:
                # 没有问题，检查推理是否正面
                has_positive = any(
                    word in thinking
                    for word in ['正确', '良好', '完善', '完整']
                )
                accuracies.append(1.0 if has_positive else 0.5)
            else:
                # 有问题，检查识别率
                recall = mentioned_issues / actual_issues if actual_issues > 0 else 0
                accuracies.append(min(recall + 0.3, 1.0))

        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    @staticmethod
    def sql_issue_detection_rate(sql_list: List[str]) -> Dict[str, float]:
        """
        计算SQL问题检测率

        Args:
            sql_list: SQL查询列表

        Returns:
            各种问题的检测率字典
        """
        classifier = ComplexityClassifier()
        issue_counts = {}
        total_count = len(sql_list)

        for sql in sql_list:
            classification = classifier.classify(sql)
            for issue in classification.issue_details:
                issue_type = issue.issue_type
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # 转换为比率
        issue_rates = {
            issue_type: count / total_count
            for issue_type, count in issue_counts.items()
        }

        return issue_rates

    @staticmethod
    def complexity_distribution(complexity_list: List[str]) -> Dict[str, float]:
        """
        计算复杂度分布

        Args:
            complexity_list: 复杂度类型列表

        Returns:
            分布字典 {类型: 比例}
        """
        if not complexity_list:
            return {}

        distribution = {}
        for complexity in complexity_list:
            distribution[complexity] = distribution.get(complexity, 0) + 1

        total = len(complexity_list)
        return {
            complexity: count / total
            for complexity, count in distribution.items()
        }

    @staticmethod
    def mean_severity_score(sql_list: List[str]) -> float:
        """
        计算平均问题严重程度

        Args:
            sql_list: SQL查询列表

        Returns:
            平均严重程度评分 (0-1)
        """
        classifier = ComplexityClassifier()
        severity_scores = []

        for sql in sql_list:
            classification = classifier.classify(sql)
            severity_scores.append(classification.severity_score)

        return sum(severity_scores) / len(severity_scores) if severity_scores else 0.0

    @staticmethod
    def has_critical_issues(sql_list: List[str]) -> float:
        """
        计算包含严重问题的SQL比例

        严重问题定义：severity_score >= 0.7

        Args:
            sql_list: SQL查询列表

        Returns:
            包含严重问题的比例 (0-1)
        """
        classifier = ComplexityClassifier()
        critical_count = 0

        for sql in sql_list:
            classification = classifier.classify(sql)
            if classification.severity_score >= 0.7:
                critical_count += 1

        return critical_count / len(sql_list) if sql_list else 0.0

    @staticmethod
    def where_clause_coverage(sql_list: List[str]) -> float:
        """
        计算包含WHERE子句的SQL比例

        Args:
            sql_list: SQL查询列表

        Returns:
            比例 (0-1)
        """
        where_count = sum(1 for sql in sql_list if 'WHERE' in sql.upper())
        return where_count / len(sql_list) if sql_list else 0.0

    @staticmethod
    def time_range_coverage(sql_list: List[str]) -> float:
        """
        计算包含时间范围的SQL比例

        Args:
            sql_list: SQL查询列表

        Returns:
            比例 (0-1)
        """
        time_keywords = ['DAY_ID', 'MONTH_ID', 'YEAR_ID', 'QUARTER_ID']
        time_count = sum(
            1 for sql in sql_list
            if any(keyword in sql.upper() for keyword in time_keywords)
        )
        return time_count / len(sql_list) if sql_list else 0.0

    @staticmethod
    def join_completeness(sql_list: List[str]) -> float:
        """
        计算JOIN完整性

        检查有JOIN的SQL是否都有ON条件

        Args:
            sql_list: SQL查询列表

        Returns:
            完整性评分 (0-1)
        """
        join_count = 0
        complete_join_count = 0

        for sql in sql_list:
            sql_upper = sql.upper()
            if 'JOIN' in sql_upper:
                join_count += 1
                # 检查是否有ON条件
                if 'ON' in sql_upper:
                    complete_join_count += 1

        return complete_join_count / join_count if join_count > 0 else 1.0

    @staticmethod
    def group_by_completeness(sql_list: List[str]) -> float:
        """
        计算GROUP BY完整性

        检查使用聚合函数的SQL是否都有GROUP BY

        Args:
            sql_list: SQL查询列表

        Returns:
            完整性评分 (0-1)
        """
        agg_functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']
        agg_count = 0
        complete_agg_count = 0

        for sql in sql_list:
            sql_upper = sql.upper()
            if any(agg in sql_upper for agg in agg_functions):
                agg_count += 1
                # 检查是否有GROUP BY
                if 'GROUP BY' in sql_upper:
                    complete_agg_count += 1

        return complete_agg_count / agg_count if agg_count > 0 else 1.0

    @staticmethod
    def calculate_all_metrics(predicted_types: List[str],
                             reference_types: List[str],
                             thinking_list: List[str],
                             sql_list: List[str]) -> Dict:
        """
        计算所有指标

        Args:
            predicted_types: 预测的类型列表
            reference_types: 参考类型列表
            thinking_list: 推理过程列表
            sql_list: SQL查询列表

        Returns:
            包含所有指标的字典
        """
        metrics = {
            'type_accuracy': Metrics.type_accuracy(predicted_types, reference_types),
            'sql_validity': Metrics.sql_validity_score(sql_list),
            'thinking_quality': Metrics.thinking_quality_score(thinking_list),
            'self_assessment_accuracy': Metrics.self_assessment_accuracy(thinking_list, sql_list),
            'mean_severity': Metrics.mean_severity_score(sql_list),
            'critical_issues_ratio': Metrics.has_critical_issues(sql_list),
            'where_clause_coverage': Metrics.where_clause_coverage(sql_list),
            'time_range_coverage': Metrics.time_range_coverage(sql_list),
            'join_completeness': Metrics.join_completeness(sql_list),
            'group_by_completeness': Metrics.group_by_completeness(sql_list),
            'complexity_distribution': Metrics.complexity_distribution(predicted_types),
            'issue_detection_rate': Metrics.sql_issue_detection_rate(sql_list),
        }

        return metrics
