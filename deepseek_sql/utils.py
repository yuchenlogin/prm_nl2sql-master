"""
DeepSeek SQL 适配模块工具函数

包含各种辅助函数和工具类
"""

import re
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_sql_from_text(text: str) -> Optional[str]:
    """
    从文本中提取SQL查询

    Args:
        text: 包含SQL的文本

    Returns:
        提取的SQL或None
    """
    # 尝试从<sql>标签中提取
    sql_match = re.search(r'<sql>(.*?)</sql>', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    # 尝试从代码块中提取
    code_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()

    # 尝试从标准SQL关键字开始提取
    sql_patterns = [
        r'(SELECT\s+.*?)(?:\n\n|\Z)',  # SELECT到下一个空段落或结束
        r'(WITH\s+.*?)(?:\n\n|\Z)',    # WITH子句
    ]

    for pattern in sql_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_thinking_from_text(text: str) -> Optional[str]:
    """
    从文本中提取推理过程

    Args:
        text: 包含推理的文本

    Returns:
        提取的推理或None
    """
    # 从<thinking>标签中提取
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL | re.IGNORECASE)
    if thinking_match:
        return thinking_match.group(1).strip()

    # 寻找反思相关的段落
    reflection_patterns = [
        r'反思[：:]\s*(.*?)(?=\n\n|\n[^\s]|\Z)',
        r'分析[：:]\s*(.*?)(?=\n\n|\n[^\s]|\Z)',
        r'推理[：:]\s*(.*?)(?=\n\n|\n[^\s]|\Z)',
    ]

    for pattern in reflection_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_self_eval_from_text(text: str) -> Optional[str]:
    """
    从文本中提取自评估

    Args:
        text: 包含自评估的文本

    Returns:
        提取的自评估或None
    """
    # 从<self_eval>标签中提取
    eval_match = re.search(r'<self_eval>(.*?)</self_eval>', text, re.DOTALL | re.IGNORECASE)
    if eval_match:
        return eval_match.group(1).strip()

    # 寻求评估相关的段落
    eval_patterns = [
        r'自评[估]?[：:]\s*(.*?)(?=\n\n|\n[^\s]|\Z)',
        r'评估[：:]\s*(.*?)(?=\n\n|\n[^\s]|\Z)',
        r'检查[：:]\s*(.*?)(?=\n\n|\n[^\s]|\Z)',
    ]

    for pattern in eval_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def normalize_sql(sql: str) -> str:
    """
    标准化SQL格式

    Args:
        sql: 原始SQL

    Returns:
        标准化后的SQL
    """
    if not sql:
        return ""

    # 移除多余空格
    sql = re.sub(r'\s+', ' ', sql)

    # 标准化关键字大小写
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
                'ON', 'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'WITH', 'AS',
                'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'NULL',
                'IS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END']

    for keyword in keywords:
        # 匹配整个单词
        pattern = r'\b' + keyword + r'\b'
        sql = re.sub(pattern, keyword, sql, flags=re.IGNORECASE)

    return sql.strip()


def calculate_sql_similarity(sql1: str, sql2: str) -> float:
    """
    计算两个SQL的相似度

    Args:
        sql1: 第一个SQL
        sql2: 第二个SQL

    Returns:
        相似度分数 (0-1)
    """
    if not sql1 and not sql2:
        return 1.0
    if not sql1 or not sql2:
        return 0.0

    # 标准化SQL
    norm_sql1 = normalize_sql(sql1).upper()
    norm_sql2 = normalize_sql(sql2).upper()

    # 完全匹配
    if norm_sql1 == norm_sql2:
        return 1.0

    # 计算编辑距离相似度（简化版本）
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    distance = levenshtein_distance(norm_sql1, norm_sql2)
    max_len = max(len(norm_sql1), len(norm_sql2))

    similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
    return max(0.0, similarity)


def validate_sql_basic(sql: str) -> Tuple[bool, List[str]]:
    """
    基本的SQL验证

    Args:
        sql: SQL查询

    Returns:
        (是否有效, 问题列表)
    """
    issues = []

    if not sql:
        issues.append("SQL is empty")
        return False, issues

    sql_upper = sql.upper().strip()

    # 检查基本结构
    if not sql_upper.startswith('SELECT'):
        issues.append("SQL must start with SELECT")
        return False, issues

    # 检查括号匹配
    if sql.count('(') != sql.count(')'):
        issues.append("Unmatched parentheses")

    # 检查引号匹配
    single_quotes = sql.count("'")
    if single_quotes % 2 != 0:
        issues.append("Unmatched single quotes")

    # 检查表名
    tables = re.findall(r'\bFROM\s+(\w+)\b', sql_upper, re.IGNORECASE)
    if not tables:
        issues.append("No table found in FROM clause")

    # 检查JOIN条件
    joins = re.findall(r'\bJOIN\s+\w+', sql_upper, re.IGNORECASE)
    ons = re.findall(r'\bON\s+', sql_upper, re.IGNORECASE)
    if joins and len(joins) != len(ons):
        issues.append("JOIN without proper ON condition")

    # 检查GROUP BY与聚合函数
    if 'GROUP BY' in sql_upper:
        select_part = sql_upper.split('FROM')[0]
        agg_functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']
        has_agg = any(func in select_part for func in agg_functions)
        if not has_agg:
            issues.append("GROUP BY without aggregation function")

    is_valid = len(issues) == 0
    return is_valid, issues


def generate_unique_id(content: str) -> str:
    """
    基于内容生成唯一ID

    Args:
        content: 内容

    Returns:
        唯一ID
    """
    return hashlib.md5(content.encode()).hexdigest()[:16]


def format_time_duration(seconds: float) -> str:
    """
    格式化时间持续时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m{seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h{int(minutes)}m{seconds:.0f}s"


def safe_json_loads(json_str: str, default=None):
    """
    安全的JSON解析

    Args:
        json_str: JSON字符串
        default: 默认值

    Returns:
        解析结果或默认值
    """
    try:
        import json
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default


def format_reward_breakdown(reward_breakdown: Dict[str, Any]) -> str:
    """
    格式化奖励分解信息

    Args:
        reward_breakdown: 奖励分解字典

    Returns:
        格式化的字符串
    """
    parts = []
    for category, data in reward_breakdown.items():
        if isinstance(data, dict) and 'score' in data and 'contribution' in data:
            score = data['score']
            contribution = data['contribution']
            parts.append(f"{category}: {score:.3f}×{data.get('weight', 0.0):.2f}={contribution:.3f}")

    return "; ".join(parts)


class ProcessTimer:
    """过程计时器"""

    def __init__(self):
        self.start_time = None
        self.checkpoints = {}

    def start(self):
        """开始计时"""
        self.start_time = datetime.now()
        logger.debug("Timer started")

    def checkpoint(self, name: str):
        """记录检查点"""
        if self.start_time is None:
            logger.warning("Timer not started")
            return

        current_time = datetime.now()
        duration = (current_time - self.start_time).total_seconds()
        self.checkpoints[name] = {
            'timestamp': current_time,
            'duration': duration
        }
        logger.debug(f"Checkpoint '{name}': {duration:.2f}s")

    def get_total_duration(self) -> float:
        """获取总持续时间"""
        if self.start_time is None:
            return 0.0

        current_time = datetime.now()
        return (current_time - self.start_time).total_seconds()

    def get_checkpoint_duration(self, name: str) -> float:
        """获取检查点持续时间"""
        return self.checkpoints.get(name, {}).get('duration', 0.0)

    def get_summary(self) -> Dict[str, Any]:
        """获取计时摘要"""
        if self.start_time is None:
            return {}

        total_duration = self.get_total_duration()
        summary = {
            'total_duration': total_duration,
            'checkpoints': self.checkpoints.copy()
        }

        # 计算各阶段的百分比
        for name, checkpoint in self.checkpoints.items():
            checkpoint['percentage'] = (checkpoint['duration'] / total_duration * 100) if total_duration > 0 else 0

        return summary


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names = {}

    def set_step_name(self, step: int, name: str):
        """设置步骤名称"""
        self.step_names[step] = name

    def advance(self, step_name: str = None) -> bool:
        """推进到下一步"""
        if self.current_step < self.total_steps:
            self.current_step += 1
            self.step_names[self.current_step] = step_name or f"Step {self.current_step}"
            logger.info(f"Progress: {self.current_step}/{self.total_steps} - {self.step_names[self.current_step]}")
            return True
        return False

    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        return (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0.0

    def is_complete(self) -> bool:
        """检查是否完成"""
        return self.current_step >= self.total_steps

    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percentage': self.get_progress_percentage(),
            'current_step_name': self.step_names.get(self.current_step, ''),
            'is_complete': self.is_complete()
        }


def benchmark_sql_generations(generations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析SQL生成基准

    Args:
        generations: 生成结果列表

    Returns:
        基准分析结果
    """
    if not generations:
        return {'error': 'No generations provided'}

    stats = {
        'total_generations': len(generations),
        'successful_generations': sum(1 for g in generations if g.get('success', False)),
        'avg_sql_length': 0,
        'avg_thinking_length': 0,
        'avg_self_eval_length': 0,
        'sql_validity_rate': 0,
        'common_issues': {}
    }

    total_sql_length = 0
    total_thinking_length = 0
    total_self_eval_length = 0
    valid_sql_count = 0
    issue_counts = {}

    for gen in generations:
        if gen.get('success', False):
            sql = gen.get('sql', '')
            thinking = gen.get('thinking', '')
            self_eval = gen.get('self_eval', '')

            total_sql_length += len(sql)
            total_thinking_length += len(thinking)
            total_self_eval_length += len(self_eval)

        # 验证SQL
        is_valid, issues = validate_sql_basic(gen.get('sql', ''))
        if is_valid:
            valid_sql_count += 1

        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    # 计算平均值
    successful_count = stats['successful_generations']
    if successful_count > 0:
        stats['avg_sql_length'] = total_sql_length / successful_count
        stats['avg_thinking_length'] = total_thinking_length / successful_count
        stats['avg_self_eval_length'] = total_self_eval_length / successful_count

    stats['sql_validity_rate'] = valid_sql_count / len(generations) if generations else 0
    stats['common_issues'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])

    return stats