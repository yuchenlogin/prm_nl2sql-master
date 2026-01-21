"""
SQL验证器

基于规则和启发式方法的SQL验证系统，包括：
1. 语法验证
2. 逻辑验证
3. 业务规则验证
4. 性能评估
"""

import re
import logging
import torch
import sqlparse
from typing import Dict, Any, List, Tuple, Optional
from .templates import SQLTemplates

# 引入现有的分类器
from classifiers.complexity_classifier import TaskTypeClassifier
from classifiers.meta_classifier import MetaClassifier

logger = logging.getLogger(__name__)


class SQLVerifier:
    """SQL验证器类"""

    def __init__(self):
        """初始化SQL验证器"""
        self.templates = SQLTemplates()
        self.task_classifier = TaskTypeClassifier()
        self.meta_classifier = MetaClassifier()

        # 验证权重配置
        self.weights = {
            'syntax': 0.3,      # 语法正确性
            'logic': 0.4,       # 逻辑正确性
            'business': 0.2,    # 业务一致性
            'performance': 0.1  # 查询效率
        }

    def verify_sql(self, query: str, sql: str, schema: str, knowledge: str,
                   model=None, tokenizer=None) -> Dict[str, Any]:
        """
        验证SQL查询

        Args:
            query: 用户问题
            sql: 生成的SQL
            schema: 数据库Schema
            knowledge: 业务知识
            model: 语言模型（可选，用于深度验证）
            tokenizer: 分词器（可选）

        Returns:
            验证结果字典
        """
        try:
            # 基于规则的快速验证
            rule_based_result = self._rule_based_verification(query, sql, schema, knowledge)

            # 如果提供了模型，进行深度验证
            if model and tokenizer:
                model_based_result = self._model_based_verification(
                    query, sql, schema, knowledge, model, tokenizer
                )
                # 结合两种验证结果
                final_result = self._combine_verification_results(
                    rule_based_result, model_based_result
                )
            else:
                final_result = rule_based_result

            return final_result

        except Exception as e:
            logger.error(f"SQL verification failed: {e}")
            return {
                'score': 0.0,
                'analysis': f"Verification error: {e}",
                'issues': ['Verification process failed'],
                'syntax_score': 0.0,
                'logic_score': 0.0,
                'business_score': 0.0,
                'performance_score': 0.0,
                'detailed_issues': {},
                'success': False
            }

    def _rule_based_verification(self, query: str, sql: str, schema: str,
                                knowledge: str) -> Dict[str, Any]:
        """
        基于规则的验证

        Args:
            query: 用户问题
            sql: SQL查询
            schema: 数据库Schema
            knowledge: 业务知识

        Returns:
            验证结果字典
        """
        result = {
            'syntax_score': 0.0,
            'logic_score': 0.0,
            'business_score': 0.0,
            'performance_score': 0.0,
            'issues': [],
            'detailed_issues': {}
        }

        # 1. 语法验证
        syntax_score, syntax_issues = self._verify_syntax(sql)
        result['syntax_score'] = syntax_score
        result['issues'].extend(syntax_issues)
        result['detailed_issues']['syntax'] = syntax_issues

        # 2. 逻辑验证
        logic_score, logic_issues = self._verify_logic(query, sql, schema)
        result['logic_score'] = logic_score
        result['issues'].extend(logic_issues)
        result['detailed_issues']['logic'] = logic_issues

        # 3. 业务验证
        business_score, business_issues = self._verify_business_rules(query, sql, knowledge)
        result['business_score'] = business_score
        result['issues'].extend(business_issues)
        result['detailed_issues']['business'] = business_issues

        # 4. 性能验证
        performance_score, performance_issues = self._verify_performance(sql)
        result['performance_score'] = performance_score
        result['issues'].extend(performance_issues)
        result['detailed_issues']['performance'] = performance_issues

        # 计算总分
        total_score = (
            result['syntax_score'] * self.weights['syntax'] +
            result['logic_score'] * self.weights['logic'] +
            result['business_score'] * self.weights['business'] +
            result['performance_score'] * self.weights['performance']
        )

        # 去重问题列表
        result['issues'] = list(set(result['issues']))

        result.update({
            'score': total_score,
            'analysis': self._generate_analysis(result),
            'method': 'rule_based',
            'success': True
        })

        return result

    def _model_based_verification(self, query: str, sql: str, schema: str,
                                 knowledge: str, model, tokenizer) -> Dict[str, Any]:
        """
        基于模型的验证

        Args:
            query: 用户问题
            sql: SQL查询
            schema: 数据库Schema
            knowledge: 业务知识
            model: 语言模型
            tokenizer: 分词器

        Returns:
            模型验证结果字典
        """
        try:
            # 构建验证提示
            prompt = self.templates.get_verification_template(
                query=query,
                sql=sql,
                schema=schema,
                knowledge=knowledge
            )

            # 生成验证响应
            response = self._generate_model_response(prompt, model, tokenizer)

            # 解析模型响应
            parsed_result = self._parse_verification_response(response)

            return parsed_result

        except Exception as e:
            logger.error(f"Model-based verification failed: {e}")
            # 回退到规则验证
            return self._rule_based_verification(query, sql, schema, knowledge)

    def _verify_syntax(self, sql: str) -> Tuple[float, List[str]]:
        """
        验证SQL语法

        Args:
            sql: SQL查询

        Returns:
            (语法分数, 问题列表)
        """
        issues = []
        score = 1.0

        try:
            # 使用sqlparse进行解析
            parsed = sqlparse.parse(sql)[0]

            # 检查基本SQL结构
            if not parsed.get_type() or parsed.get_type() == 'UNKNOWN':
                issues.append("Invalid SQL structure")
                score -= 0.3

            # 检查括号匹配
            if sql.count('(') != sql.count(')'):
                issues.append("Unmatched parentheses")
                score -= 0.2

            # 检查引号匹配
            single_quotes = sql.count("'")
            if single_quotes % 2 != 0:
                issues.append("Unmatched single quotes")
                score -= 0.2

            double_quotes = sql.count('"')
            if double_quotes % 2 != 0:
                issues.append("Unmatched double quotes")
                score -= 0.1

            # 检查常见的语法错误
            if 'JOIN' in sql.upper() and 'ON' not in sql.upper():
                issues.append("JOIN without ON condition")
                score -= 0.2

            if 'GROUP BY' in sql.upper() and 'HAVING' not in sql.upper():
                # 检查GROUP BY相关的聚合函数
                select_part = sql.upper().split('FROM')[0].split('SELECT')[1]
                agg_functions = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']
                has_agg = any(func in select_part for func in agg_functions)
                grouping_cols = select_part.count(',')
                if not has_agg and grouping_cols > 0:
                    issues.append("GROUP BY without aggregation")
                    score -= 0.15

        except Exception as e:
            issues.append(f"SQL parsing error: {e}")
            score = 0.0

        return max(0.0, score), issues

    def _verify_logic(self, query: str, sql: str, schema: str) -> Tuple[float, List[str]]:
        """
        验证SQL逻辑

        Args:
            query: 用户问题
            sql: SQL查询
            schema: 数据库Schema

        Returns:
            (逻辑分数, 问题列表)
        """
        issues = []
        score = 1.0

        # 使用现有的分类器进行验证
        try:
            classification_result = self.task_classifier.classify(sql, query)

            # 检查严重程度
            if classification_result.severity_score > 0.7:
                issues.extend([issue.issue_type for issue in classification_result.issue_details])
                score -= classification_result.severity_score * 0.5

            # 使用元分类器检查SQL有效性
            sql_validity = self.meta_classifier._check_sql_validity(sql)
            if sql_validity < 0.8:
                issues.append("SQL logic validation failed")
                score -= (1.0 - sql_validity) * 0.3

        except Exception as e:
            logger.warning(f"Logic verification using classifiers failed: {e}")

        # 自定义逻辑检查
        logic_issues = self._custom_logic_checks(query, sql, schema)
        issues.extend(logic_issues)
        score -= len(logic_issues) * 0.1

        return max(0.0, score), issues

    def _verify_business_rules(self, query: str, sql: str, knowledge: str) -> Tuple[float, List[str]]:
        """
        验证业务规则

        Args:
            query: 用户问题
            sql: SQL查询
            knowledge: 业务知识

        Returns:
            (业务分数, 问题列表)
        """
        issues = []
        score = 1.0

        # 时间范围检查
        time_range_issues = self._check_time_range(query, sql, knowledge)
        issues.extend(time_range_issues)
        score -= len(time_range_issues) * 0.15

        # 指标定义检查
        metric_issues = self._check_metric_definitions(query, sql, knowledge)
        issues.extend(metric_issues)
        score -= len(metric_issues) * 0.1

        return max(0.0, score), issues

    def _verify_performance(self, sql: str) -> Tuple[float, List[str]]:
        """
        验证查询性能

        Args:
            sql: SQL查询

        Returns:
            (性能分数, 问题列表)
        """
        issues = []
        score = 1.0

        sql_upper = sql.upper()

        # 检查SELECT *
        if 'SELECT *' in sql_upper:
            issues.append("Using SELECT * may impact performance")
            score -= 0.1

        # 检查缺少WHERE条件
        if 'WHERE' not in sql_upper and 'JOIN' in sql_upper:
            issues.append("Missing WHERE clause in join query")
            score -= 0.1

        # 检查ORDER BY without LIMIT
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            issues.append("ORDER BY without LIMIT may impact performance")
            score -= 0.05

        # 检查子查询使用
        if sql_upper.count('(SELECT') > 2:
            issues.append("Multiple subqueries may impact performance")
            score -= 0.1

        return max(0.0, score), issues

    def _custom_logic_checks(self, query: str, sql: str, schema: str) -> List[str]:
        """
        自定义逻辑检查

        Args:
            query: 用户问题
            sql: SQL查询
            schema: 数据库Schema

        Returns:
            问题列表
        """
        issues = []

        # 检查表引用一致性
        tables_in_sql = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql)
        # 这里可以添加更复杂的表名提取逻辑

        # 检查字段名是否存在
        # 这里可以解析Schema并检查字段名

        # 检查JOIN逻辑
        if 'JOIN' in sql.upper():
            # 检查是否有必要的连接条件
            if 'ON' not in sql.upper():
                issues.append("JOIN without proper ON condition")

        return issues

    def _check_time_range(self, query: str, sql: str, knowledge: str) -> List[str]:
        """
        检查时间范围处理

        Args:
            query: 用户问题
            sql: SQL查询
            knowledge: 业务知识

        Returns:
            时间范围相关的问题列表
        """
        issues = []

        # 检查问题中的时间关键词
        time_keywords = ['今年', '去年', '本月', '上月', '最近', '过去', 'Q', '季度', '月', '周', '天']
        has_time_keyword = any(keyword in query for keyword in time_keywords)

        if has_time_keyword:
            # 检查SQL中是否有时间条件
            sql_upper = sql.upper()
            time_fields = ['DAY_ID', 'MONTH_ID', 'YEAR_ID', 'QUARTER_ID', 'WEEK_ID']
            has_time_condition = any(field in sql_upper for field in time_fields)

            if not has_time_condition:
                issues.append("Query involves time but SQL lacks time range condition")

        return issues

    def _check_metric_definitions(self, query: str, sql: str, knowledge: str) -> List[str]:
        """
        检查指标定义

        Args:
            query: 用户问题
            sql: SQL查询
            knowledge: 业务知识

        Returns:
            指标相关的问题列表
        """
        issues = []

        # 检查净线索量特殊处理
        if '净' in query or '线索' in query:
            sql_upper = sql.upper()
            if 'IS_NET_LEADS' not in sql_upper:
                issues.append("Net leads query should filter by is_net_leads")

        # 检查去重处理
        if 'COUNT(' in sql.upper() and 'DISTINCT' not in sql.upper():
            issues.append("Count operation may need DISTINCT for leads counting")

        return issues

    def _generate_analysis(self, result: Dict[str, Any]) -> str:
        """生成验证分析文本"""
        analysis_parts = []

        if result['syntax_score'] < 0.8:
            analysis_parts.append("Syntax issues detected")

        if result['logic_score'] < 0.8:
            analysis_parts.append("Logic problems found")

        if result['business_score'] < 0.8:
            analysis_parts.append("Business rule violations")

        if result['performance_score'] < 0.8:
            analysis_parts.append("Performance concerns")

        if not analysis_parts:
            return "SQL appears to be correct and well-formed"

        return "Issues found: " + ", ".join(analysis_parts)

    def _generate_model_response(self, prompt: str, model, tokenizer) -> str:
        """生成模型响应"""
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.3,  # 验证时使用较低温度
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """解析模型验证响应"""
        result = {
            'score': 0.0,
            'analysis': "",
            'issues': [],
            'success': True
        }

        try:
            # 提取分析部分
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
            if analysis_match:
                result['analysis'] = analysis_match.group(1).strip()

            # 提取分数
            score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
            if score_match:
                try:
                    result['score'] = float(score_match.group(1).strip())
                except ValueError:
                    result['score'] = 0.0

            # 提取问题列表
            issues_match = re.search(r'<issues>(.*?)</issues>', response, re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1).strip()
                if issues_text.lower() != 'none':
                    result['issues'] = [issue.strip() for issue in issues_text.split(',') if issue.strip()]

        except Exception as e:
            result['success'] = False
            result['issues'] = [f"Parsing error: {e}"]

        return result

    def _combine_verification_results(self, rule_result: Dict[str, Any],
                                    model_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        结合基于规则和模型的验证结果

        Args:
            rule_result: 规则验证结果
            model_result: 模型验证结果

        Returns:
            结合后的验证结果
        """
        # 简单的加权平均，优先信任模型的深度分析
        combined_score = rule_result['score'] * 0.4 + model_result['score'] * 0.6

        # 合并问题列表
        combined_issues = list(set(rule_result['issues'] + model_result['issues']))

        # 使用模型的详细分析
        combined_result = model_result.copy()
        combined_result.update({
            'score': combined_score,
            'issues': combined_issues,
            'rule_based_score': rule_result['score'],
            'model_based_score': model_result['score'],
            'method': 'combined'
        })

        return combined_result