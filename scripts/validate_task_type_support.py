"""
验证八种任务类型支持的正确性
检查所有相关代码文件是否正确支持八种标准任务类型
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

from data.data_loader import NL2SQLDataLoader
from classifiers.complexity_classifier import TaskTypeClassifier
from reward.reward_model import ProcessRewardModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 八种标准任务类型
EXPECTED_TASK_TYPES = {
    "SQL",  # 简单思考直接输出SQL
    "多步推理",  # 多步骤思考，输出可能带有CTE等的复杂SQL
    "反思",  # 将输入的错误SQL更正
    "歧义澄清",  # 用户问题包含歧义点，触发模型思考
    "维度拒识",  # 用户问题包含查询不支持的维度时模型拒绝回答
    "维度退化",  # 维表退化到事实表时仍支持查询
    "指标拒识",  # 用户问题包含查询不支持的指标时模型拒绝回答
    "追问"  # 用户问题不满足查询的必备要求
}


class TaskTypeValidator:
    """任务类型验证器"""

    def __init__(self, data_files: List[str]):
        """
        初始化验证器

        Args:
            data_files: 待验证的数据文件列表
        """
        self.data_files = data_files
        self.loader = NL2SQLDataLoader()
        self.classifier = TaskTypeClassifier()
        self.reward_model = ProcessRewardModel()
        self.validation_results = {}

    def validate_all(self) -> Dict:
        """
        执行所有验证测试

        Returns:
            验证结果字典
        """
        logger.info("="*60)
        logger.info("开始验证八种任务类型支持")
        logger.info("="*60)

        # 1. 验证数据文件中的任务类型
        self._validate_data_file_task_types()

        # 2. 验证分类器对八种类型的支持
        self._validate_classifier_support()

        # 3. 验证奖励模型对八种类型的支持
        self._validate_reward_model_support()

        # 4. 验证数据加载器的类型处理
        self._validate_data_loader_support()

        # 5. 生成验证报告
        self._generate_validation_report()

        return self.validation_results

    def _validate_data_file_task_types(self):
        """验证数据文件中的任务类型"""
        logger.info("\n1. 验证数据文件中的任务类型分布")
        logger.info("-"*50)

        task_type_counts = {}
        all_found_types = set()

        for data_file in self.data_files:
            logger.info(f"\n检查文件: {data_file}")
            examples = self.loader.load(data_file, use_cache=False)

            for example in examples:
                task_type = example.task_type
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
                all_found_types.add(task_type)

        # 检查是否有未预期的任务类型
        unexpected_types = all_found_types - EXPECTED_TASK_TYPES
        missing_types = EXPECTED_TASK_TYPES - all_found_types

        self.validation_results['data_file_validation'] = {
            'task_type_counts': task_type_counts,
            'found_types': list(all_found_types),
            'unexpected_types': list(unexpected_types),
            'missing_types': list(missing_types),
            'is_valid': len(unexpected_types) == 0
        }

        logger.info(f"发现的任务类型: {sorted(all_found_types)}")
        logger.info(f"任务类型分布: {task_type_counts}")

        if unexpected_types:
            logger.warning(f"发现未预期的任务类型: {sorted(unexpected_types)}")

        if missing_types:
            logger.warning(f"缺失的任务类型: {sorted(missing_types)}")

    def _validate_classifier_support(self):
        """验证分类器对八种类型的支持"""
        logger.info("\n2. 验证分类器对八种类型的支持")
        logger.info("-"*50)

        classifier_issues = []

        # 检查分类器支持的类型
        for task_type in EXPECTED_TASK_TYPES:
            # 创建测试样本
            test_sql = "SELECT 1" if task_type in ["SQL", "多步推理", "反思", "维度退化"] else ""
            test_query = f"测试{task_type}类问题"

            try:
                result = self.classifier.classify(test_sql, test_query, task_type)

                if result.task_type != task_type:
                    classifier_issues.append(f"{task_type}: 预期类型不匹配 (预期: {task_type}, 实际: {result.task_type})")

                logger.info(f"✓ {task_type}: 分类成功，任务类型={result.task_type}")
            except Exception as e:
                classifier_issues.append(f"{task_type}: 分类失败 - {str(e)}")
                logger.error(f"✗ {task_type}: 分类失败 - {str(e)}")

        self.validation_results['classifier_validation'] = {
            'issues': classifier_issues,
            'is_valid': len(classifier_issues) == 0
        }

        if classifier_issues:
            logger.error(f"\n分类器问题 ({len(classifier_issues)}个):")
            for issue in classifier_issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("\n✓ 所有任务类型分类通过")

    def _validate_reward_model_support(self):
        """验证奖励模型对八种类型的支持"""
        logger.info("\n3. 验证奖励模型对八种类型的支持")
        logger.info("-"*50)

        reward_issues = []

        # 检查奖励模型支持的类型
        for task_type in EXPECTED_TASK_TYPES:
            test_sql = "SELECT 1" if task_type in ["SQL", "多步推理", "反思", "维度退化"] else ""
            test_thinking = f"这是{task_type}类任务的推理过程"
            test_query = f"测试{task_type}类问题"

            try:
                reward_dict = self.reward_model.compute_reward(
                    generated_sql=test_sql,
                    predicted_type=task_type,
                    thinking=test_thinking,
                    reference_type=task_type,
                    reference_sql=test_sql,
                    query=test_query
                )

                # 检查奖励值范围
                total_reward = reward_dict['total_reward']
                if not (0 <= total_reward <= 1):
                    reward_issues.append(f"{task_type}: 奖励值超出范围 [0,1] - {total_reward}")

                # 检查可训练性标记
                is_trainable = reward_dict['is_trainable']
                expected_trainable = task_type in ProcessRewardModel.TRAINABLE_TASK_TYPES
                if is_trainable != expected_trainable:
                    reward_issues.append(f"{task_type}: 可训练性标记错误 (预期: {expected_trainable}, 实际: {is_trainable})")

                logger.info(f"✓ {task_type}: 奖励计算成功，总奖励={total_reward:.4f}，可训练={is_trainable}")
            except Exception as e:
                reward_issues.append(f"{task_type}: 奖励计算失败 - {str(e)}")
                logger.error(f"✗ {task_type}: 奖励计算失败 - {str(e)}")

        self.validation_results['reward_model_validation'] = {
            'issues': reward_issues,
            'is_valid': len(reward_issues) == 0
        }

        if reward_issues:
            logger.error(f"\n奖励模型问题 ({len(reward_issues)}个):")
            for issue in reward_issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("\n✓ 所有任务类型奖励计算通过")

    def _validate_data_loader_support(self):
        """验证数据加载器的类型处理"""
        logger.info("\n4. 验证数据加载器的类型处理")
        logger.info("-"*50)

        loader_issues = []
        trainable_examples = 0
        non_trainable_examples = 0

        for data_file in self.data_files:
            examples = self.loader.load(data_file, use_cache=False)

            for example in examples:
                task_type = example.task_type

                # 检查复杂度类型兼容性
                complexity_type = example.complexity_type

                # 对于可训练类型，复杂度类型应该是sql或多步推理
                if example.is_trainable:
                    if task_type == "多步推理":
                        expected_complexity = "多步推理"
                    else:
                        expected_complexity = "sql"

                    if complexity_type != expected_complexity:
                        loader_issues.append(f"{task_type}: 复杂度类型不匹配 (预期: {expected_complexity}, 实际: {complexity_type})")

                # 检查可训练性
                if example.is_trainable:
                    trainable_examples += 1
                    if task_type in ProcessRewardModel.NON_TRAINABLE_TASK_TYPES:
                        loader_issues.append(f"{task_type}: 应该是不可训练的但被标记为可训练")
                else:
                    non_trainable_examples += 1
                    if task_type in ProcessRewardModel.TRAINABLE_TASK_TYPES:
                        logger.warning(f"{task_type}: 应该是可训练的但被标记为不可训练")

        self.validation_results['data_loader_validation'] = {
            'issues': loader_issues[:10],  # 只显示前10个错误，避免输出过长
            'total_issues': len(loader_issues),
            'trainable_examples': trainable_examples,
            'non_trainable_examples': non_trainable_examples,
            'is_valid': len(loader_issues) == 0
        }

        logger.info(f"可训练样本: {trainable_examples}")
        logger.info(f"非训练样本: {non_trainable_examples}")

        if loader_issues:
            logger.error(f"\n数据加载器问题 ({len(loader_issues)}个):")
            for issue in loader_issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("\n✓ 数据加载器类型处理通过")

    def _generate_validation_report(self):
        """生成验证报告"""
        logger.info("\n5. 生成验证报告")
        logger.info("="*50)

        # 汇总验证结果
        all_validations = [
            self.validation_results['data_file_validation']['is_valid'],
            self.validation_results['classifier_validation']['is_valid'],
            self.validation_results['reward_model_validation']['is_valid'],
            self.validation_results['data_loader_validation']['is_valid']
        ]

        overall_valid = all(all_validations)

        self.validation_results['overall_valid'] = overall_valid
        self.validation_results['summary'] = {
            'data_files_checked': len(self.data_files),
            'expected_task_types': len(EXPECTED_TASK_TYPES),
            'validations_passed': sum(all_validations),
            'validations_total': len(all_validations)
        }

        # 保存报告
        report_file = "./outputs/validation_report.json"
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, ensure_ascii=False, indent=2)

        logger.info(f"\n验证报告已保存到: {report_file}\n")

        # 打印汇总
        logger.info("验证汇总:")
        logger.info(f"- 数据文件验证: {'✓ 通过' if all_validations[0] else '✗ 失败'}")
        logger.info(f"- 分类器验证: {'✓ 通过' if all_validations[1] else '✗ 失败'}")
        logger.info(f"- 奖励模型验证: {'✓ 通过' if all_validations[2] else '✗ 失败'}")
        logger.info(f"- 数据加载器验证: {'✓ 通过' if all_validations[3] else '✗ 失败'}")

        if overall_valid:
            logger.info("\n🎉 所有验证通过！八种任务类型支持正常。")
        else:
            logger.error("\n❌ 验证失败！请检查上述问题并修复。")


def main():
    """主函数"""
    # 要验证的数据文件
    data_files = [
        "./data/nl2_sql_cold_start_sft_all_train_swift_9501_1231.json",
        "./data/nl2_sql_cold_start_sft_all_test_swift_830_1231.json"
    ]

    # 检查文件是否存在
    existing_files = []
    for file_path in data_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
        else:
            logger.warning(f"数据文件不存在: {file_path}")

    if not existing_files:
        logger.error("没有找到可用的数据文件，验证终止")
        return

    # 执行验证
    validator = TaskTypeValidator(existing_files)
    results = validator.validate_all()

    # 根据验证结果设置退出码
    exit_code = 0 if results['overall_valid'] else 1
    exit(exit_code)


if __name__ == "__main__":
    main()