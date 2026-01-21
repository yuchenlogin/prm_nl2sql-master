"""
SQL复杂度分类器
分类SQL的复杂度并检测常见问题
"""

import logging
import re
from enum import Enum
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    SQL = "SQL"  # 简单思考直接输出SQL
    MULTI_STEP = "多步推理"  # 多步骤思考，输出可能带有CTE等的复杂SQL
    REFLECTION = "反思"  # 将输入的错误SQL更正
    AMBIGUITY = "歧义澄清"  # 用户问题包含歧义点，触发模型思考
    DIMENSION_REFUSE = "维度拒识"  # 用户问题包含查询不支持的维度时模型拒绝回答
    DIMENSION_DEGRADE = "维度退化"  # 维表退化到事实表时仍支持查询
    METRIC_REFUSE = "指标拒识"  # 用户问题包含查询不支持的指标时模型拒绝回答
    FOLLOW_UP = "追问"  # 用户问题不满足查询的必备要求


@dataclass
class SQLIssue:
    """SQL问题数据类"""
    issue_type: str
    severity: float  # 0-1，1表示最严重
    description: str
    line_number: Optional[int] = None


@dataclass
class ClassificationResult:
    """分类结果"""
    task_type: str
    complexity: str  # 保留兼容性：sql或 多步推理
    issues: Dict[str, bool] = field(default_factory=dict)
    issue_details: list = field(default_factory=list)
    severity_score: float = 0.0
    confidence: float = 1.0


class TaskTypeClassifier:
    """
    任务类型分类器
    支持8种任务类型的分类：
    1. SQL - 简单思考直接输出SQL
    2. 多步推理 - 多步骤思考，输出可能带有CTE等的复杂SQL
    3. 反思 - 将输入的错误SQL更正
    4. 歧义澄清 - 用户问题包含歧义点，触发模型思考
    5. 维度拒识 - 用户问题包含查询不支持的维度时模型拒绝回答
    6. 维度退化 - 维表退化到事实表时仍支持查询
    7. 指标拒识 - 用户问题包含查询不支持的指标时模型拒绝回答
    8. 追问 - 用户问题不满足查询的必备要求

    功能：
    1. 分类任务类型
    2. 检测11种常见问题
    3. 计算问题严重程度
    """

    # 常见问题检查规则
    ISSUE_RULES = {
        'missing_where': {
            'pattern': lambda sql: 'WHERE' not in sql.upper(),
            'severity': 0.3,
            'description': '缺少WHERE条件'
        },
        'missing_time_range': {
            'pattern': lambda sql: (
                'WHERE' in sql.upper() and
                not any(t in sql.upper() for t in ['DAY_ID', 'MONTH_ID', 'YEAR_ID', 'QUARTER_ID'])
            ),
            'severity': 0.4,
            'description': '缺少时间范围（NL2SQL关键）'
        },
        'incorrect_join': {
            'pattern': lambda sql: 'JOIN' in sql.upper() and 'ON' not in sql.upper(),
            'severity': 0.5,
            'description': 'JOIN缺少ON条件'
        },
        'missing_join_condition': {
            'pattern': lambda sql: (
                'LEFT JOIN' in sql.upper() or 'INNER JOIN' in sql.upper()
            ) and sql.upper().count('ON') < sql.upper().count('JOIN'),
            'severity': 0.45,
            'description': '某些JOIN缺少ON条件'
        },
        'inconsistent_alias': {
            'pattern': lambda sql: (
                ('t1' in sql or 't2' in sql or 't3' in sql) and
                ('FROM' in sql.upper() and not re.search(r'\bAS\s+t\d', sql, re.IGNORECASE))
            ),
            'severity': 0.25,
            'description': '表别名不一致'
        },
        'missing_group_by': {
            'pattern': lambda sql: (
                'GROUP BY' not in sql.upper() and
                any(agg in sql.upper() for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])
            ),
            'severity': 0.35,
            'description': '使用聚合函数但未有GROUP BY'
        },
        'unclosed_parenthesis': {
            'pattern': lambda sql: sql.count('(') != sql.count(')'),
            'severity': 0.8,
            'description': '括号不匹配'
        },
        'empty_in_clause': {
            'pattern': lambda sql: re.search(r'\bIN\s*\(\s*\)', sql, re.IGNORECASE),
            'severity': 0.7,
            'description': 'IN子句为空'
        },
        'invalid_date_format': {
            'pattern': lambda sql: (
                ('DATE' in sql.upper() or 'DAY_ID' in sql.upper()) and
                re.search(r"'\d{2}-\d{2}-\d{4}'", sql)  # 错误的日期格式
            ),
            'severity': 0.5,
            'description': '日期格式不正确'
        },
        'multiple_tables_no_join': {
            'pattern': lambda sql: (
                sql.upper().count('FROM') > 1 and
                'JOIN' not in sql.upper()
            ),
            'severity': 0.4,
            'description': '多个表但未使用JOIN'
        },
        'order_by_without_limit': {
            'pattern': lambda sql: (
                'ORDER BY' in sql.upper() and
                'LIMIT' not in sql.upper()
            ),
            'severity': 0.2,
            'description': '有ORDER BY但无LIMIT'
        },
    }

    @classmethod
    def classify(cls, sql: str, query: str = None, task_type_hint: str = None) -> ClassificationResult:
        """
        分类任务类型和SQL复杂度

        Args:
            sql: SQL查询字符串
            query: 用户问题（可选，用于判断任务类型）
            task_type_hint: 任务类型提示（可选）

        Returns:
            ClassificationResult
        """
        if not sql or not isinstance(sql, str):
            return ClassificationResult(
                task_type=task_type_hint or "SQL",
                complexity="sql",
                issues={},
                severity_score=1.0,
                confidence=0.0
            )

        # 检测问题
        issues = {}
        issue_details = []

        for issue_type, rule in cls.ISSUE_RULES.items():
            try:
                has_issue = rule['pattern'](sql)
                issues[issue_type] = has_issue
                if has_issue:
                    issue_details.append(SQLIssue(
                        issue_type=issue_type,
                        severity=rule['severity'],
                        description=rule['description']
                    ))
            except Exception as e:
                logger.warning(f"检查{issue_type}时出错: {e}")

        # 计算任务类型
        task_type = cls._compute_task_type(sql, query, task_type_hint)

        # 计算复杂度（保持兼容性）
        complexity = cls._compute_complexity(sql)

        # 计算严重程度
        severity_score = cls._compute_severity(issues)

        # 计算置信度
        confidence = max(0.0, 1.0 - severity_score * 0.7)

        return ClassificationResult(
            task_type=task_type,
            complexity=complexity,
            issues=issues,
            issue_details=issue_details,
            severity_score=severity_score,
            confidence=confidence
        )

    @staticmethod
    def _compute_task_type(sql: str, query: str = None, task_type_hint: str = None) -> str:
        """
        计算任务类型

        规则：
        - 如果有提示，优先使用提示
        - 根据查询内容判断是否为拒识或澄清类型
        - 根据SQL复杂度判断是SQL还是多步推理
        """
        if task_type_hint:
            return task_type_hint

        # 如果没有SQL或SQL无效，可能是拒识类型
        if not sql or not sql.strip():
            if query:
                if any(keyword in query for keyword in ['不明', '不清楚', '哪个', '哪种', '什么']):
                    return "歧义澄清"
                if any(keyword in query for keyword in ['不存在', '不支持', '没有']):
                    return "维度拒识"
            return "SQL"

        # 根据SQL内容判断复杂度
        complexity_score = TaskTypeClassifier._compute_complexity(sql)
        if complexity_score == "多步推理":
            return "多步推理"
        else:
            return "SQL"

    @staticmethod
    def _compute_complexity(sql: str) -> str:
        """
        计算SQL复杂度

        规则：
        - JOIN数 >= 2 或 CTE数 >= 1 或 UNION数 >= 1 → 多步推理
        - 其他 → 简单SQL
        """
        sql_upper = sql.upper()

        join_count = sql_upper.count('JOIN')
        cte_count = sql_upper.count('WITH')
        union_count = sql_upper.count('UNION')
        subquery_count = len(re.findall(r'\(.*SELECT.*\)', sql, re.IGNORECASE))

        complexity_score = (
            join_count * 2 +
            cte_count * 3 +
            union_count * 2.5 +
            subquery_count * 1.5
        )

        if complexity_score > 3:
            return "多步推理"
        else:
            return "sql"

    @staticmethod
    def _compute_severity(issues: Dict[str, bool]) -> float:
        """
        计算问题的总体严重程度

        Args:
            issues: 问题字典 {issue_type: bool}

        Returns:
            严重程度 (0-1)
        """
        severity_weights = {
            'unclosed_parenthesis': 0.8,
            'incorrect_join': 0.5,
            'missing_time_range': 0.4,
            'missing_group_by': 0.35,
            'invalid_date_format': 0.3,
            'missing_where': 0.3,
            'inconsistent_alias': 0.25,
            'multiple_tables_no_join': 0.25,
            'missing_join_condition': 0.2,
            'empty_in_clause': 0.2,
            'order_by_without_limit': 0.1,
        }

        total_severity = 0.0
        for issue_type, has_issue in issues.items():
            if has_issue:
                total_severity += severity_weights.get(issue_type, 0.1)

        return min(total_severity, 1.0)  # 限制在0-1

    @classmethod
    def get_detailed_report(cls, sql: str, query: str = None, task_type_hint: str = None) -> Dict:
        """
        获取详细的分析报告

        Args:
            sql: SQL查询字符串
            query: 用户问题（可选）
            task_type_hint: 任务类型提示（可选）

        Returns:
            详细报告字典
        """
        result = cls.classify(sql, query, task_type_hint)

        detected_issues = [
            {
                'type': issue.issue_type,
                'severity': issue.severity,
                'description': issue.description
            }
            for issue in result.issue_details
        ]

        return {
            'task_type': result.task_type,
            'complexity': result.complexity,
            'severity_score': result.severity_score,
            'confidence': result.confidence,
            'total_issues': len(detected_issues),
            'detected_issues': detected_issues,
            'is_valid': result.severity_score < 0.5,
            'recommendation': cls._get_recommendation(result)
        }

    @staticmethod
    def _get_recommendation(result: ClassificationResult) -> str:
        """获取改进建议"""
        if result.severity_score >= 0.7:
            return "⚠️ 严重问题，需要重新审视整个SQL逻辑"
        elif result.severity_score >= 0.4:
            return "⚠️ 存在中等问题，建议检查JOIN、时间范围和聚合逻辑"
        elif result.severity_score >= 0.2:
            return "✓ 轻微问题，可能影响性能或可读性"
        else:
            return "✅ SQL质量良好"


# Optional imports
from typing import Optional
