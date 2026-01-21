"""
数据加载和处理模块
处理NL2SQL数据集的加载、验证、转换
支持新的任务场景数据格式
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import re

logger = logging.getLogger(__name__)


class NL2SQLExample:
    """单个NL2SQL样本"""

    def __init__(self,
                 query: str,
                 response: str,
                 thinking: str,
                 sql: str,
                 task_type: str,
                 example_output: str,
                 sample_id: Optional[str] = None):
        self.query = query
        self.response = response  # 原始响应
        self.thinking = thinking
        self.sql = sql
        self.task_type = task_type  # 任务类型：SQL、多步推理、反思、歧义澄清、维度拒识等
        self.example_output = example_output  # 示例输出
        self.sample_id = sample_id or str(hash((query, response)))

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'query': self.query,
            'response': self.response,
            'thinking': self.thinking,
            'sql': self.sql,
            'task_type': self.task_type,
            'example_output': self.example_output,
            'sample_id': self.sample_id
        }

    def __repr__(self):
        return (f"NL2SQLExample(type={self.task_type}, "
                f"query_len={len(self.query)}, has_sql={bool(self.sql)})")

    @property
    def is_trainable(self) -> bool:
        """判断样本是否可用于训练"""
        # 不可训练的类型：拒识类
        non_trainable_types = ["维度拒识", "指标拒识", "歧义澄清", "追问"]

        # 只有SQL、多步推理、反思、维度退化是可训练的
        return self.task_type not in non_trainable_types

    @property
    def complexity_type(self) -> str:
        """兼容旧版本的复杂度类型，基于新的八种任务类型"""
        # 将八种任务类型映射为两种复杂度类型
        if self.task_type == "多步推理":
            return "多步推理"
        elif self.is_trainable:
            # 可训练类型中，除了多步推理，其他都归为sql
            return "sql"
        else:
            # 非训练类型保持原样
            return self.task_type


class NL2SQLDataLoader:
    """
    NL2SQL数据加载器
    支持新的任务场景数据格式：query + response (包含think和<answer>) + type + example_output
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.examples: List[NL2SQLExample] = []

    def load(self, data_file: str, use_cache: bool = True) -> List[NL2SQLExample]:
        """
        加载数据文件

        Args:
            data_file: JSON数据文件路径
            use_cache: 是否使用缓存

        Returns:
            NL2SQLExample列表
        """
        cache_file = self.cache_dir / f"{Path(data_file).stem}.pkl"

        # 尝试从缓存加载
        if use_cache and cache_file.exists():
            logger.info(f"从缓存加载: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.examples = pickle.load(f)
            return self.examples

        # 从原始文件加载
        logger.info(f"从文件加载: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.examples = []
        for idx, item in enumerate(raw_data):
            try:
                example = self._parse_item(item, idx)
                if example:
                    self.examples.append(example)
            except Exception as e:
                logger.warning(f"解析第{idx}条数据失败: {e}")
                continue

        logger.info(f"成功加载 {len(self.examples)} 条数据")

        # 保存缓存
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.examples, f)
            logger.info(f"缓存已保存: {cache_file}")

        return self.examples

    def _parse_item(self, item: Dict, idx: int) -> Optional[NL2SQLExample]:
        """解析单个数据项 - 支持新的任务场景数据格式"""
        try:
            query = item.get('query', '')
            response = item.get('response', '')
            task_type = item.get('type', 'SQL')  # 默认为SQL类型
            example_output = item.get('example_output', '')

            # 标准化任务类型名称
            task_type = self._normalize_task_type(task_type)

            # 验证必需字段
            if not query or not response:
                logger.warning(f"第{idx}条数据缺少query或response")
                return None

            # 提取think部分
            thinking = self._extract_section(response, 'think')

            # 提取<answer>部分
            sql = self._extract_section(response, 'answer')

            # 对于拒识类型，answer可能不是SQL
            if task_type in ["维度拒识", "指标拒识", "歧义澄清", "追问"]:
                # 提取澄清或拒识的回答
                if not sql:
                    # 尝试从example_output获取答案
                    sql = example_output or ""
                # 对于歧义澄清和追问，可能需要特殊的响应处理
                if task_type in ["歧义澄清", "追问"] and example_output:
                    sql = example_output
            elif task_type == "反思":
                # 反思类型，answer是修正后的SQL
                if not sql:
                    sql = example_output or ""
            elif task_type == "维度退化":
                # 维度退化，正常提取SQL
                if not sql:
                    logger.warning(f"第{idx}条数据为维度退化类型但无法提取SQL")
                    return None

            return NL2SQLExample(
                query=query,
                response=response,
                thinking=thinking,
                sql=sql,
                task_type=task_type,
                example_output=example_output,
                sample_id=f"sample_{idx}"
            )

        except Exception as e:
            logger.error(f"解析第{idx}条数据异常: {e}")
            return None

    @staticmethod
    def _normalize_task_type(task_type: str) -> str:
        """
        标准化任务类型名称

        Args:
            task_type: 原始任务类型

        Returns:
            标准化后的任务类型
        """
        # 任务类型映射表
        task_type_mapping = {
            # 数据文件中的类型 -> 代码中的标准类型
            'sql': 'SQL',
            '反思数据': '反思',
            '追问_必备约束': '追问',
            'SQL': 'SQL',
            '多步推理': '多步推理',
            '反思': '反思',
            '歧义澄清': '歧义澄清',
            '维度拒识': '维度拒识',
            '维度退化': '维度退化',
            '指标拒识': '指标拒识',
            '追问': '追问'
        }

        return task_type_mapping.get(task_type, task_type)

    @staticmethod
    def _extract_section(text: str, section: str) -> str:
        """
        从文本中提取指定标签的内容

        Args:
            text: 原始文本
            section: 标签名（think或answer）

        Returns:
            标签内容
        """
        start_tag = f"<{section}>"
        end_tag = f"</{section}>"

        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            return ""

        content = text[start_idx + len(start_tag):end_idx].strip()
        return content

    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.examples:
            return {}

        # 统计各种任务类型
        task_type_counts = {}
        trainable_count = 0
        for example in self.examples:
            task_type = example.task_type
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            if example.is_trainable:
                trainable_count += 1

        stats = {
            'total_samples': len(self.examples),
            'trainable_samples': trainable_count,
            'non_trainable_samples': len(self.examples) - trainable_count,
            'avg_query_length': sum(len(e.query) for e in self.examples) / len(self.examples),
            'avg_sql_length': sum(len(e.sql) for e in self.examples) / len(self.examples),
            'avg_thinking_length': sum(len(e.thinking) for e in self.examples if e.thinking) / max(1, sum(1 for e in self.examples if e.thinking)),
            'task_type_distribution': {k: f"{v/len(self.examples)*100:.1f}%" for k, v in task_type_counts.items()}
        }

        # 传统统计（兼容旧版本）
        stats['simple_sql'] = stats['trainable_samples'] - task_type_counts.get('多步推理', 0)
        stats['multi_step'] = task_type_counts.get('多步推理', 0)
        stats['type_distribution'] = {
            'sql': f"{stats['simple_sql']/len(self.examples)*100:.1f}%",
            '多步推理': f"{stats['multi_step']/len(self.examples)*100:.1f}%"
        }

        return stats

    def split_train_test(self,
                        val_split: float = 0.1,
                        random_seed: int = 42) -> Tuple[List[NL2SQLExample], List[NL2SQLExample]]:
        """
        分割训练集和验证集

        Args:
            val_split: 验证集比例
            random_seed: 随机种子

        Returns:
            (train_examples, val_examples)
        """
        import random
        random.seed(random_seed)

        shuffled = self.examples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_split))
        return shuffled[:split_idx], shuffled[split_idx:]

    def save_processed(self, output_dir: str):
        """保存处理后的数据为JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        data = [ex.to_dict() for ex in self.examples]

        with open(output_path / 'all_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存到: {output_path / 'all_data.json'}")

    def get_trainable_examples(self) -> List[NL2SQLExample]:
        """获取可训练的样本（排除拒识和反思类型）"""
        return [ex for ex in self.examples if ex.is_trainable]

    def get_examples_by_type(self, task_type: str) -> List[NL2SQLExample]:
        """根据任务类型获取样本"""
        return [ex for ex in self.examples if ex.task_type == task_type]


def validate_data(data_file: str) -> Dict:
    """
    快速验证数据文件的完整性

    Args:
        data_file: JSON数据文件路径

    Returns:
        验证报告
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    report = {
        'total_count': len(data),
        'valid_count': 0,
        'invalid_count': 0,
        'missing_query': 0,
        'missing_response': 0,
        'type_distribution': {},
        'invalid_format': []
    }

    task_type_counts = {}

    for idx, item in enumerate(data):
        if 'query' not in item or 'response' not in item:
            report['invalid_count'] += 1
            if 'query' not in item:
                report['missing_query'] += 1
            if 'response' not in item:
                report['missing_response'] += 1
            continue

        # 统计任务类型
        task_type = item.get('type', 'SQL')
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        # 检查think和<answer>
        response = item['response']
        if '<think>' not in response or '<answer>' not in response:
            report['invalid_format'].append(idx)
            report['invalid_count'] += 1
            continue

        report['valid_count'] += 1

    report['validity_rate'] = f"{report['valid_count']/len(data)*100:.1f}%"
    report['type_distribution'] = {k: f"{v/len(data)*100:.1f}%" for k, v in task_type_counts.items()}

    return report