"""
数据准备脚本
验证、统计、转换数据
"""

import json
import logging
from pathlib import Path
from typing import Dict
from data.data_loader import NL2SQLDataLoader, validate_data

logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./outputs/logs/data_preparation.log'),
            logging.StreamHandler()
        ]
    )


def validate_dataset(data_file: str) -> Dict:
    """
    验证数据集

    Args:
        data_file: 数据文件路径

    Returns:
        验证报告
    """
    logger.info(f"验证数据文件: {data_file}")

    report = validate_data(data_file)

    logger.info("=" * 50)
    logger.info("数据验证报告")
    logger.info("=" * 50)
    logger.info(f"总数据条数: {report['total_count']}")
    logger.info(f"有效条数: {report['valid_count']}")
    logger.info(f"无效条数: {report['invalid_count']}")
    logger.info(f"缺少query: {report['missing_query']}")
    logger.info(f"缺少response: {report['missing_response']}")
    logger.info(f"有效率: {report['validity_rate']}")

    if report['invalid_format']:
        logger.warning(f"格式无效的条目索引: {report['invalid_format'][:10]}...")

    return report


def collect_statistics(data_file: str) -> Dict:
    """
    收集数据统计信息

    Args:
        data_file: 数据文件路径

    Returns:
        统计信息字典
    """
    logger.info(f"收集数据统计: {data_file}")

    loader = NL2SQLDataLoader()
    examples = loader.load(data_file)

    stats = loader.get_statistics()

    logger.info("=" * 50)
    logger.info("数据统计信息")
    logger.info("=" * 50)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    return stats


def analyze_complexity_distribution(data_file: str) -> Dict:
    """
    分析复杂度分布

    Args:
        data_file: 数据文件路径

    Returns:
        复杂度分布字典
    """
    logger.info(f"分析复杂度分布: {data_file}")

    loader = NL2SQLDataLoader()
    examples = loader.load(data_file)

    simple_count = sum(1 for e in examples if e.task_type == 'SQL')
    multi_step_count = sum(1 for e in examples if e.task_type == '多步推理')
    reflection_count = sum(1 for e in examples if e.task_type == '反思')
    ambiguity_count = sum(1 for e in examples if e.task_type == '歧义澄清')
    dimension_refuse_count = sum(1 for e in examples if e.task_type == '维度拒识')
    dimension_degrade_count = sum(1 for e in examples if e.task_type == '维度退化')
    metric_refuse_count = sum(1 for e in examples if e.task_type == '指标拒识')
    follow_up_count = sum(1 for e in examples if e.task_type == '追问')
    total = len(examples)

    distribution = {
        'sql': simple_count,
        'multi_step': multi_step_count,
        'reflection': reflection_count,
        'ambiguity': ambiguity_count,
        'dimension_refuse': dimension_refuse_count,
        'dimension_degrade': dimension_degrade_count,
        'metric_refuse': metric_refuse_count,
        'follow_up': follow_up_count,
        'sql_ratio': simple_count / total if total > 0 else 0,
        'multi_step_ratio': multi_step_count / total if total > 0 else 0,
        'reflection_ratio': reflection_count / total if total > 0 else 0,
        'ambiguity_ratio': ambiguity_count / total if total > 0 else 0,
        'dimension_refuse_ratio': dimension_refuse_count / total if total > 0 else 0,
        'dimension_degrade_ratio': dimension_degrade_count / total if total > 0 else 0,
        'metric_refuse_ratio': metric_refuse_count / total if total > 0 else 0,
        'follow_up_ratio': follow_up_count / total if total > 0 else 0,
        'total': total
    }

    logger.info("=" * 50)
    logger.info("任务类型分布")
    logger.info("=" * 50)
    logger.info(f"SQL: {simple_count} ({distribution['sql_ratio']*100:.1f}%)")
    logger.info(f"多步推理: {multi_step_count} ({distribution['multi_step_ratio']*100:.1f}%)")
    logger.info(f"反思: {reflection_count} ({distribution['reflection_ratio']*100:.1f}%)")
    logger.info(f"歧义澄清: {ambiguity_count} ({distribution['ambiguity_ratio']*100:.1f}%)")
    logger.info(f"维度拒识: {dimension_refuse_count} ({distribution['dimension_refuse_ratio']*100:.1f}%)")
    logger.info(f"维度退化: {dimension_degrade_count} ({distribution['dimension_degrade_ratio']*100:.1f}%)")
    logger.info(f"指标拒识: {metric_refuse_count} ({distribution['metric_refuse_ratio']*100:.1f}%)")
    logger.info(f"追问: {follow_up_count} ({distribution['follow_up_ratio']*100:.1f}%)")

    return distribution


def split_dataset(data_file: str,
                  train_ratio: float = 0.9,
                  output_dir: str = "./data/processed") -> Dict:
    """
    分割数据集为训练集和验证集

    Args:
        data_file: 数据文件路径
        train_ratio: 训练集比例
        output_dir: 输出目录

    Returns:
        分割结果字典
    """
    logger.info(f"分割数据集: {data_file}")

    loader = NL2SQLDataLoader()
    examples = loader.load(data_file)

    # 分割
    train_examples, val_examples = loader.split_train_test(
        val_split=1.0 - train_ratio
    )

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存处理后的数据
    train_data = [ex.to_dict() for ex in train_examples]
    val_data = [ex.to_dict() for ex in val_examples]

    train_file = output_path / 'train.json'
    val_file = output_path / 'val.json'

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    result = {
        'train_count': len(train_examples),
        'val_count': len(val_examples),
        'train_file': str(train_file),
        'val_file': str(val_file),
        'train_ratio': len(train_examples) / len(examples),
        'val_ratio': len(val_examples) / len(examples)
    }

    logger.info("=" * 50)
    logger.info("数据集分割结果")
    logger.info("=" * 50)
    logger.info(f"训练集: {result['train_count']} ({result['train_ratio']*100:.1f}%)")
    logger.info(f"验证集: {result['val_count']} ({result['val_ratio']*100:.1f}%)")
    logger.info(f"训练文件: {train_file}")
    logger.info(f"验证文件: {val_file}")

    return result


def analyze_query_lengths(data_file: str) -> Dict:
    """
    分析查询长度分布

    Args:
        data_file: 数据文件路径

    Returns:
        长度分析字典
    """
    logger.info(f"分析查询长度: {data_file}")

    loader = NL2SQLDataLoader()
    examples = loader.load(data_file)

    query_lengths = [len(e.query) for e in examples]
    sql_lengths = [len(e.sql) for e in examples]
    thinking_lengths = [len(e.thinking) for e in examples if e.thinking]

    analysis = {
        'query_length': {
            'min': min(query_lengths),
            'max': max(query_lengths),
            'avg': sum(query_lengths) / len(query_lengths),
            'median': sorted(query_lengths)[len(query_lengths)//2]
        },
        'sql_length': {
            'min': min(sql_lengths),
            'max': max(sql_lengths),
            'avg': sum(sql_lengths) / len(sql_lengths),
            'median': sorted(sql_lengths)[len(sql_lengths)//2]
        },
        'thinking_length': {
            'min': min(thinking_lengths) if thinking_lengths else 0,
            'max': max(thinking_lengths) if thinking_lengths else 0,
            'avg': sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0,
            'median': sorted(thinking_lengths)[len(thinking_lengths)//2] if thinking_lengths else 0
        }
    }

    logger.info("=" * 50)
    logger.info("长度分析")
    logger.info("=" * 50)
    logger.info(f"查询长度 - 平均: {analysis['query_length']['avg']:.0f}, "
                f"范围: [{analysis['query_length']['min']}, {analysis['query_length']['max']}]")
    logger.info(f"SQL长度 - 平均: {analysis['sql_length']['avg']:.0f}, "
                f"范围: [{analysis['sql_length']['min']}, {analysis['sql_length']['max']}]")
    logger.info(f"推理长度 - 平均: {analysis['thinking_length']['avg']:.0f}, "
                f"范围: [{analysis['thinking_length']['min']}, {analysis['thinking_length']['max']}]")

    return analysis


def generate_preparation_report(train_file: str, test_file: str, output_file: str = "./outputs/data_preparation_report.json"):
    """
    生成数据准备报告

    Args:
        train_file: 训练文件
        test_file: 测试文件
        output_file: 输出文件
    """
    logger.info("=" * 70)
    logger.info("NL2SQL数据准备报告")
    logger.info("=" * 70)

    report = {
        'timestamp': Path.cwd(),
        'train_file': {
            'path': train_file,
            'validation': validate_dataset(train_file),
            'statistics': collect_statistics(train_file),
            'complexity_distribution': analyze_complexity_distribution(train_file),
            'length_analysis': analyze_query_lengths(train_file)
        },
        'test_file': {
            'path': test_file,
            'validation': validate_dataset(test_file),
            'statistics': collect_statistics(test_file),
            'complexity_distribution': analyze_complexity_distribution(test_file),
            'length_analysis': analyze_query_lengths(test_file)
        }
    }

    # 保存报告
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, float):
            return round(obj, 4)
        else:
            return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(make_serializable(report), f, ensure_ascii=False, indent=2)

    logger.info(f"\n报告已保存: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="数据准备脚本")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="训练文件路径"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="测试文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="输出目录"
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging()

    # 生成报告
    report_file = Path(args.output_dir) / "data_preparation_report.json"
    generate_preparation_report(
        train_file=args.train_file,
        test_file=args.test_file,
        output_file=str(report_file)
    )

    logger.info("\n数据准备完成！")


if __name__ == "__main__":
    main()
