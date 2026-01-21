"""
SQL元验证器

负责评估SQL验证器的质量，确保验证结果的可靠性和一致性
包括：验证逻辑评估、一致性检查、置信度计算
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from .templates import SQLTemplates

logger = logging.getLogger(__name__)


class SQLMetaVerifier:
    """SQL元验证器类"""

    def __init__(self):
        """初始化SQL元验证器"""
        self.templates = SQLTemplates()

        # 元验证标准
        self.consistency_threshold = 0.7  # 一致性阈值
        self.confidence_weights = {
            'multiple_verifications': 0.4,  # 多次验证的一致性
            'logic_reasoning': 0.3,         # 推理逻辑合理性
            'issue_detection': 0.2,         # 问题检测质量
            'score_calibration': 0.1        # 分数校准准确性
        }

    def meta_verify(self, query: str, sql: str, verification_results: List[Dict[str, Any]],
                   schema: str, knowledge: str, model=None, tokenizer=None) -> Dict[str, Any]:
        """
        执行元验证

        Args:
            query: 用户问题
            sql: SQL查询
            verification_results: 多次验证的结果列表
            schema: 数据库Schema
            knowledge: 业务知识
            model: 语言模型（可选）
            tokenizer: 分词器（可选）

        Returns:
            元验证结果字典
        """
        try:
            # 1. 基于规则的元验证
            rule_based_meta = self._rule_based_meta_verification(
                verification_results, query, sql
            )

            # 2. 基于模型的深度元验证
            if model and tokenizer and len(verification_results) > 0:
                model_based_meta = self._model_based_meta_verification(
                    query, sql, verification_results[0], schema, knowledge, model, tokenizer
                )
                # 两种结果融合
                final_result = self._combine_meta_results(rule_based_meta, model_based_meta)
            else:
                final_result = rule_based_meta

            return final_result

        except Exception as e:
            logger.error(f"Meta verification failed: {e}")
            return {
                'adjusted_score': 0.0,
                'confidence': 'low',
                'meta_analysis': f"Meta verification error: {e}",
                'consistency_score': 0.0,
                'reliability_score': 0.0,
                'success': False
            }

    def _rule_based_meta_verification(self, verification_results: List[Dict[str, Any]],
                                     query: str, sql: str) -> Dict[str, Any]:
        """
        基于规则的元验证

        Args:
            verification_results: 验证结果列表
            query: 用户问题
            sql: SQL查询

        Returns:
            规则元验证结果
        """
        if not verification_results:
            return {
                'adjusted_score': 0.0,
                'confidence': 'low',
                'meta_analysis': "No verification results available",
                'consistency_score': 0.0,
                'reliability_score': 0.0,
                'method': 'rule_based',
                'success': False
            }

        # 1. 计算一致性分数
        consistency_score = self._calculate_consistency_score(verification_results)

        # 2. 评估验证器可靠性
        reliability_score = self._assess_verifier_reliability(verification_results, query, sql)

        # 3. 检测异常验证结果
        outliers = self._detect_outlier_verifications(verification_results)
        filtered_results = [r for i, r in enumerate(verification_results) if i not in outliers]

        # 4. 计算调整后的分数
        if filtered_results:
            avg_score = sum(r['score'] for r in filtered_results) / len(filtered_results)
        else:
            avg_score = sum(r['score'] for r in verification_results) / len(verification_results)

        # 5. 基于一致性和可靠性调整分数
        adjustment_factor = (consistency_score + reliability_score) / 2
        adjusted_score = avg_score * adjustment_factor + (1 - adjustment_factor) * 0.5  # 回归到中性分数

        # 6. 计算整体置信度
        confidence = self._calculate_confidence(consistency_score, reliability_score, len(verification_results))

        # 7. 生成元分析
        meta_analysis = self._generate_meta_analysis(
            consistency_score, reliability_score, outliers, len(verification_results)
        )

        return {
            'adjusted_score': max(0.0, min(1.0, adjusted_score)),
            'confidence': confidence,
            'meta_analysis': meta_analysis,
            'consistency_score': consistency_score,
            'reliability_score': reliability_score,
            'outlier_indices': outliers,
            'verification_count': len(verification_results),
            'average_score': avg_score,
            'method': 'rule_based',
            'success': True
        }

    def _model_based_meta_verification(self, query: str, sql: str,
                                      verification_result: Dict[str, Any],
                                      schema: str, knowledge: str,
                                      model, tokenizer) -> Dict[str, Any]:
        """
        基于模型的元验证

        Args:
            query: 用户问题
            sql: SQL查询
            verification_result: 单个验证结果
            schema: 数据库Schema
            knowledge: 业务知识
            model: 语言模型
            tokenizer: 分词器

        Returns:
            模型元验证结果
        """
        try:
            # 构建元验证提示
            prompt = self.templates.get_meta_verification_template(
                query=query,
                sql=sql,
                verifier_score=str(verification_result.get('score', 0.0)),
                verifier_analysis=verification_result.get('analysis', ''),
                issues=', '.join(verification_result.get('issues', [])),
                schema=schema,
                knowledge=knowledge
            )

            # 生成元验证响应
            response = self._generate_model_response(prompt, model, tokenizer)

            # 解析响应
            parsed_result = self._parse_meta_verification_response(response)

            return parsed_result

        except Exception as e:
            logger.error(f"Model-based meta verification failed: {e}")
            return {
                'adjusted_score': verification_result.get('score', 0.0),
                'confidence': 'medium',
                'meta_analysis': f"Model verification failed, using original score: {e}",
                'method': 'rule_based_fallback',
                'success': False
            }

    def _calculate_consistency_score(self, verification_results: List[Dict[str, Any]]) -> float:
        """
        计算验证结果的一致性分数

        Args:
            verification_results: 验证结果列表

        Returns:
            一致性分数 (0-1)
        """
        if len(verification_results) < 2:
            return 1.0  # 单个结果默认一致

        scores = [r['score'] for r in verification_results]

        # 计算分数的标准差
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5

        # 标准差越小，一致性越高
        consistency = max(0.0, 1.0 - std_dev * 2)  # 2是经验系数

        # 考虑分类一致性（如果有的话）
        issue_lists = [set(r.get('issues', [])) for r in verification_results]
        if all(issue_lists):
            # 计算问题检测的重叠度
            common_issues = set.intersection(*issue_lists)
            all_issues = set.union(*issue_lists)

            if all_issues:
                issue_consistency = len(common_issues) / len(all_issues)
                consistency = (consistency + issue_consistency) / 2

        return consistency

    def _assess_verifier_reliability(self, verification_results: List[Dict[str, Any]],
                                    query: str, sql: str) -> float:
        """
        评估验证器的可靠性

        Args:
            verification_results: 验证结果列表
            query: 用户问题
            sql: SQL查询

        Returns:
            可靠性分数 (0-1)
        """
        reliability_factors = []

        for result in verification_results:
            factor = 1.0

            # 检查验证是否成功
            if not result.get('success', True):
                factor *= 0.5

            # 检查分析质量
            analysis = result.get('analysis', '')
            if len(analysis) < 20 or len(analysis.split()) < 5:
                factor *= 0.8  # 分析太简短

            # 检查问题检测的合理性
            issues = result.get('issues', [])
            if 'error' in str(issues).lower():
                factor *= 0.6  # 包含错误信息说明验证有问题

            # 检查分数是否在合理范围
            score = result.get('score', 0.0)
            if not (0.0 <= score <= 1.0):
                factor *= 0.7  # 分数超出范围

            reliability_factors.append(factor)

        # 返回平均可靠性
        return sum(reliability_factors) / len(reliability_factors)

    def _detect_outlier_verifications(self, verification_results: List[Dict[str, Any]]) -> List[int]:
        """
        检测异常的验证结果

        Args:
            verification_results: 验证结果列表

        Returns:
            异常结果索引列表
        """
        if len(verification_results) < 3:
            return []  # 样本太少不检测异常

        scores = [r.get('score', 0.0) for r in verification_results]
        mean_score = sum(scores) / len(scores)
        std_dev = sum((s - mean_score) ** 2 for s in scores) ** 0.5

        outliers = []
        threshold = 1.5  # 1.5个标准差之外认为是异常

        for i, score in enumerate(scores):
            if abs(score - mean_score) > threshold * std_dev:
                outliers.append(i)

        return outliers

    def _calculate_confidence(self, consistency_score: float, reliability_score: float,
                            verification_count: int) -> str:
        """
        计算整体置信度

        Args:
            consistency_score: 一致性分数
            reliability_score: 可靠性分数
            verification_count: 验证次数

        Returns:
            置信度等级：high/medium/low
        """
        # 综合评分
        composite_score = (consistency_score + reliability_score) / 2

        # 验证次数加分
        if verification_count >= 3:
            composite_score += 0.1
        elif verification_count == 1:
            composite_score -= 0.2

        # 确定置信度等级
        if composite_score >= 0.8:
            return 'high'
        elif composite_score >= 0.5:
            return 'medium'
        else:
            return 'low'

    def _generate_meta_analysis(self, consistency_score: float, reliability_score: float,
                               outliers: List[int], verification_count: int) -> str:
        """
        生成元分析文本

        Args:
            consistency_score: 一致性分数
            reliability_score: 可靠性分数
            outliers: 异常索引
            verification_count: 验证次数

        Returns:
            元分析文本
        """
        analysis_parts = []

        # 一致性分析
        if consistency_score >= 0.8:
            analysis_parts.append("验证结果高度一致")
        elif consistency_score >= 0.5:
            analysis_parts.append("验证结果基本一致")
        else:
            analysis_parts.append("验证结果存在较大分歧")

        # 可靠性分析
        if reliability_score >= 0.8:
            analysis_parts.append("验证器表现可靠")
        elif reliability_score >= 0.5:
            analysis_parts.append("验证器表现一般")
        else:
            analysis_parts.append("验证器可靠性有待提升")

        # 异常情况
        if outliers:
            analysis_parts.append(f"检测到{len(outliers)}个异常验证结果")

        # 验证次数分析
        if verification_count >= 3:
            analysis_parts.append("基于多次验证，结果可信")
        elif verification_count == 1:
            analysis_parts.append("基于单次验证，建议增加验证次数")

        return "; ".join(analysis_parts)

    def _generate_model_response(self, prompt: str, model, tokenizer) -> str:
        """生成模型响应"""
        import torch

        inputs = tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.2,  # 元验证使用更低温度
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _parse_meta_verification_response(self, response: str) -> Dict[str, Any]:
        """解析元验证响应"""
        result = {
            'adjusted_score': 0.0,
            'confidence': 'medium',
            'meta_analysis': "",
            'success': True
        }

        try:
            # 提取元分析
            meta_analysis_match = re.search(r'<meta_analysis>(.*?)</meta_analysis>', response, re.DOTALL)
            if meta_analysis_match:
                result['meta_analysis'] = meta_analysis_match.group(1).strip()

            # 提取调整后的分数
            score_match = re.search(r'<adjusted_score>(.*?)</adjusted_score>', response, re.DOTALL)
            if score_match:
                try:
                    score_text = score_match.group(1).strip()
                    if score_text.lower() != 'unchanged':
                        result['adjusted_score'] = float(score_text)
                except ValueError:
                    pass  # 保持默认值

            # 提取置信度
            confidence_match = re.search(r'<confidence>(.*?)</confidence>', response, re.DOTALL)
            if confidence_match:
                result['confidence'] = confidence_match.group(1).strip().lower()

        except Exception as e:
            result['success'] = False
            result['meta_analysis'] = f"Parsing error: {e}"

        return result

    def _combine_meta_results(self, rule_result: Dict[str, Any],
                             model_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        结合规则和模型的元验证结果

        Args:
            rule_result: 规则元验证结果
            model_result: 模型元验证结果

        Returns:
            结合后的元验证结果
        """
        combined = rule_result.copy()

        # 分数融合：模型权重稍高
        combined['adjusted_score'] = (
            rule_result['adjusted_score'] * 0.4 + model_result['adjusted_score'] * 0.6
        )

        # 置信度融合
        confidence_map = {'high': 0.8, 'medium': 0.5, 'low': 0.2}
        rule_conf_val = confidence_map.get(rule_result['confidence'], 0.5)
        model_conf_val = confidence_map.get(model_result['confidence'], 0.5)
        combined_conf_val = (rule_conf_val * 0.4 + model_conf_val * 0.6)

        # 转换回置信度等级
        if combined_conf_val >= 0.65:
            combined['confidence'] = 'high'
        elif combined_conf_val >= 0.35:
            combined['confidence'] = 'medium'
        else:
            combined['confidence'] = 'low'

        # 合并分析
        combined['meta_analysis'] = f"[规则] {rule_result['meta_analysis']}; [模型] {model_result['meta_analysis']}"

        combined.update({
            'method': 'combined',
            'rule_confidence': rule_result['confidence'],
            'model_confidence': model_result['confidence']
        })

        return combined