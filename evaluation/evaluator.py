"""
评估模块
对微调后的模型进行评估
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data.data_loader import NL2SQLDataLoader
from generator.sql_generator import SQLGenerator
from evaluation.metrics import Metrics
from generator.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class Evaluator:
    """评估器"""

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 batch_size: int = 8):
        """
        初始化评估器

        Args:
            model_path: 模型路径
            device: 设备
            batch_size: 批处理大小
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        # 初始化生成器
        logger.info(f"加载模型: {model_path}")
        self.generator = SQLGenerator(
            model_name=model_path,
            device=device,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_new_tokens=1024
        )

        # 初始化数据加载器
        self.data_loader = NL2SQLDataLoader()

    def evaluate(self,
                 test_file: str,
                 num_samples: Optional[int] = None) -> Dict:
        """
        评估模型

        Args:
            test_file: 测试文件路径
            num_samples: 评估样本数（None表示全部）

        Returns:
            评估结果字典
        """
        logger.info("=" * 50)
        logger.info("开始评估")
        logger.info("=" * 50)

        # 加载测试数据
        test_examples = self.data_loader.load(test_file)
        if num_samples and num_samples > 0:
            test_examples = test_examples[:num_samples]

        logger.info(f"加载了 {len(test_examples)} 条测试数据")

        # 生成预测
        predictions = self._generate_predictions(test_examples)

        # 计算指标
        metrics = self._compute_metrics(predictions, test_examples)

        # 生成详细报告
        report = self._generate_report(metrics, predictions, test_examples)

        logger.info("=" * 50)
        logger.info("评估完成")
        logger.info("=" * 50)

        return report

    def _generate_predictions(self, examples: List) -> List[Dict]:
        """
        生成预测

        Args:
            examples: 例子列表

        Returns:
            预测列表
        """
        logger.info("生成预测...")
        predictions = []

        schema = PromptTemplates.get_schema_context()

        for idx, example in enumerate(tqdm(examples, desc="生成预测")):
            try:
                result = self.generator.generate(
                    question=example.query,
                    schema=schema
                )

                if result['success']:
                    predictions.append({
                        'query': example.query,
                        'predicted_sql': result['sql'],
                        'predicted_type': result['complexity_type'],
                        'thinking': result['thinking'],
                        'reference_sql': example.sql,
                        'reference_type': example.complexity_type,
                        'reference_thinking': example.thinking,
                        'success': True,
                        'classification_confidence': result['classification']['confidence'],
                        'verification_confidence': result['verification']['confidence'],
                    })
                else:
                    predictions.append({
                        'query': example.query,
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'reference_type': example.complexity_type
                    })
            except Exception as e:
                logger.warning(f"生成预测失败 (sample {idx}): {e}")
                predictions.append({
                    'query': example.query,
                    'success': False,
                    'error': str(e),
                    'reference_type': example.complexity_type
                })

        logger.info(f"成功生成 {sum(1 for p in predictions if p['success'])} 个预测")
        return predictions

    def _compute_metrics(self, predictions: List[Dict], examples: List) -> Dict:
        """
        计算指标

        Args:
            predictions: 预测列表
            examples: 例子列表

        Returns:
            指标字典
        """
        logger.info("计算指标...")

        # 过滤成功的预测
        successful = [p for p in predictions if p['success']]
        if not successful:
            logger.error("没有成功的预测")
            return {}

        predicted_types = [p['predicted_type'] for p in successful]
        reference_types = [p['reference_type'] for p in successful]
        thinking_list = [p['thinking'] for p in successful]
        sql_list = [p['predicted_sql'] for p in successful]

        # 计算所有指标
        metrics = Metrics.calculate_all_metrics(
            predicted_types=predicted_types,
            reference_types=reference_types,
            thinking_list=thinking_list,
            sql_list=sql_list
        )

        # 添加成功率
        metrics['success_rate'] = len(successful) / len(predictions)

        return metrics

    def _generate_report(self, metrics: Dict, predictions: List[Dict], examples: List) -> Dict:
        """
        生成详细报告

        Args:
            metrics: 指标字典
            predictions: 预测列表
            examples: 例子列表

        Returns:
            报告字典
        """
        logger.info("生成报告...")

        successful = [p for p in predictions if p['success']]
        failed = [p for p in predictions if not p['success']]

        report = {
            'summary': {
                'total_samples': len(predictions),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': metrics.get('success_rate', 0),
            },
            'metrics': metrics,
            'examples': self._sample_predictions(successful, sample_size=5),
            'error_analysis': self._analyze_errors(failed),
        }

        return report

    @staticmethod
    def _sample_predictions(predictions: List[Dict], sample_size: int = 5) -> List[Dict]:
        """
        抽样预测

        Args:
            predictions: 预测列表
            sample_size: 样本大小

        Returns:
            样本列表
        """
        if not predictions:
            return []

        step = max(1, len(predictions) // sample_size)
        samples = []

        for i in range(0, len(predictions), step):
            if len(samples) >= sample_size:
                break
            samples.append(predictions[i])

        return samples

    @staticmethod
    def _analyze_errors(failed: List[Dict]) -> Dict:
        """
        分析错误

        Args:
            failed: 失败的预测列表

        Returns:
            错误分析字典
        """
        if not failed:
            return {'total': 0}

        error_types = {}
        for pred in failed:
            error = pred.get('error', 'Unknown')
            error_types[error] = error_types.get(error, 0) + 1

        return {
            'total': len(failed),
            'error_distribution': error_types
        }

    def save_report(self, report: Dict, output_path: str):
        """
        保存报告

        Args:
            report: 报告字典
            output_path: 输出路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为JSON可序列化格式
        report_json = self._make_serializable(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_json, f, ensure_ascii=False, indent=2)

        logger.info(f"报告已保存: {output_file}")

    @staticmethod
    def _make_serializable(obj):
        """转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: Evaluator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [Evaluator._make_serializable(item) for item in obj]
        elif isinstance(obj, (float, torch.Tensor)):
            if isinstance(obj, torch.Tensor):
                return float(obj)
            return round(obj, 4)
        else:
            return obj

    def compare_with_baseline(self,
                             baseline_report: Dict,
                             current_report: Dict) -> Dict:
        """
        与基线模型对比

        Args:
            baseline_report: 基线报告
            current_report: 当前报告

        Returns:
            对比结果
        """
        comparison = {
            'improvements': {},
            'regressions': {},
            'unchanged': {}
        }

        baseline_metrics = baseline_report.get('metrics', {})
        current_metrics = current_report.get('metrics', {})

        for metric_name in baseline_metrics:
            if metric_name in current_metrics:
                baseline_val = baseline_metrics[metric_name]
                current_val = current_metrics[metric_name]

                if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                    improvement = current_val - baseline_val
                    improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0

                    if improvement > 0.01:
                        comparison['improvements'][metric_name] = {
                            'baseline': baseline_val,
                            'current': current_val,
                            'improvement': improvement,
                            'improvement_pct': improvement_pct
                        }
                    elif improvement < -0.01:
                        comparison['regressions'][metric_name] = {
                            'baseline': baseline_val,
                            'current': current_val,
                            'regression': -improvement,
                            'regression_pct': -improvement_pct
                        }
                    else:
                        comparison['unchanged'][metric_name] = {
                            'value': current_val
                        }

        return comparison


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="NL2SQL模型评估脚本")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="测试文件路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="评估样本数"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/evaluation_report.json",
        help="输出报告路径"
    )

    args = parser.parse_args()

    # 创建评估器并评估
    evaluator = Evaluator(model_path=args.model)
    report = evaluator.evaluate(
        test_file=args.test_file,
        num_samples=args.num_samples
    )

    # 保存报告
    evaluator.save_report(report, args.output)

    # 打印摘要
    print("\n" + "=" * 50)
    print("评估摘要")
    print("=" * 50)
    summary = report.get('summary', {})
    for key, value in summary.items():
        print(f"{key}: {value}")

    metrics = report.get('metrics', {})
    print("\n主要指标:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
