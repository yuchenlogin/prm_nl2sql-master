"""
SQL生成器

负责生成SQL查询和自评估，包括：
1. SQL生成逻辑
2. 自评估机制
3. 结果解析
"""

import re
import logging
import torch
from typing import Dict, Any, Optional, Tuple
from .templates import SQLTemplates

logger = logging.getLogger(__name__)


class SQLGenerator:
    """SQL生成器类"""

    def __init__(self, model, tokenizer):
        """
        初始化SQL生成器

        Args:
            model: 语言模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.templates = SQLTemplates()

    def generate_sql_with_self_eval(self, query: str, schema: str, knowledge: str,
                                   examples: str = "", temperature: float = 0.7,
                                   max_new_tokens: int = 1024) -> Dict[str, Any]:
        """
        生成SQL查询和自评估

        Args:
            query: 用户问题
            schema: 数据库Schema
            knowledge: 业务知识
            examples: 示例（可选）
            temperature: 生成温度
            max_new_tokens: 最大生成长度

        Returns:
            包含生成结果的字典：
            {
                'thinking': str,
                'sql': str,
                'self_eval': str,
                'full_response': str,
                'success': bool,
                'error': str
            }
        """
        try:
            # 构建生成提示
            prompt = self.templates.get_generation_template(
                query=query,
                schema=schema,
                knowledge=knowledge,
                examples=examples
            )

            # 生成响应
            response = self._generate_response(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

            # 解析响应
            parsed_result = self._parse_generation_response(response)

            return parsed_result

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                'thinking': "",
                'sql': "",
                'self_eval': "",
                'full_response': "",
                'success': False,
                'error': str(e)
            }

    def refine_sql(self, query: str, original_sql: str, verification_feedback: str,
                  average_score: float, schema: str, knowledge: str,
                  temperature: float = 0.6) -> Dict[str, Any]:
        """
        基于验证反馈细化SQL

        Args:
            query: 用户问题
            original_sql: 原始SQL
            verification_feedback: 验证反馈
            average_score: 平均评分
            schema: 数据库Schema
            knowledge: 业务知识
            temperature: 生成温度

        Returns:
            细化后的结果字典
        """
        try:
            # 构建细化提示
            prompt = self.templates.get_refinement_template(
                query=query,
                original_sql=original_sql,
                verification_feedback=verification_feedback,
                average_score=average_score
            )

            # 生成响应
            response = self._generate_response(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=1024
            )

            # 解析细化响应
            parsed_result = self._parse_refinement_response(response)

            return parsed_result

        except Exception as e:
            logger.error(f"SQL refinement failed: {e}")
            return {
                'refined_thinking': "",
                'refined_sql': "",
                'refined_self_eval': "",
                'full_response': "",
                'success': False,
                'error': str(e)
            }

    def _generate_response(self, prompt: str, temperature: float,
                           max_new_tokens: int) -> str:
        """
        生成模型响应

        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_new_tokens: 最大生成长度

        Returns:
            模型响应文本
        """
        # Tokenize输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def _parse_generation_response(self, response: str) -> Dict[str, Any]:
        """
        解析生成响应

        Args:
            response: 模型完整响应

        Returns:
            解析后的结果字典
        """
        result = {
            'thinking': "",
            'sql': "",
            'self_eval': "",
            'full_response': response,
            'success': True,
            'error': ""
        }

        try:
            # 提取thinking部分
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
            if thinking_match:
                result['thinking'] = thinking_match.group(1).strip()
            else:
                result['success'] = False
                result['error'] = "No <thinking> section found"

            # 提取SQL部分
            sql_match = re.search(r'<sql>(.*?)</sql>', response, re.DOTALL)
            if sql_match:
                result['sql'] = sql_match.group(1).strip()
            else:
                result['success'] = False
                result['error'] = "No <sql> section found"

            # 提取自评估部分
            self_eval_match = re.search(r'<self_eval>(.*?)</self_eval>', response, re.DOTALL)
            if self_eval_match:
                result['self_eval'] = self_eval_match.group(1).strip()
            else:
                result['success'] = False
                result['error'] = "No <self_eval> section found"

        except Exception as e:
            result['success'] = False
            result['error'] = f"Parsing error: {e}"

        return result

    def _parse_refinement_response(self, response: str) -> Dict[str, Any]:
        """
        解析细化响应

        Args:
            response: 模型细化响应

        Returns:
            解析后的细化结果字典
        """
        result = {
            'refined_thinking': "",
            'refined_sql': "",
            'refined_self_eval': "",
            'full_response': response,
            'success': True,
            'error': ""
        }

        try:
            # 提取细化后的thinking
            thinking_match = re.search(r'<refined_thinking>(.*?)</refined_thinking>', response, re.DOTALL)
            if thinking_match:
                result['refined_thinking'] = thinking_match.group(1).strip()

            # 提取细化后的SQL
            sql_match = re.search(r'<refined_sql>(.*?)</refined_sql>', response, re.DOTALL)
            if sql_match:
                result['refined_sql'] = sql_match.group(1).strip()

            # 提取细化后的自评估
            self_eval_match = re.search(r'<refined_self_eval>(.*?)</refined_self_eval>', response, re.DOTALL)
            if self_eval_match:
                result['refined_self_eval'] = self_eval_match.group(1).strip()

            # 检查是否所有部分都成功提取
            if not all([result['refined_thinking'], result['refined_sql'], result['refined_self_eval']]):
                result['success'] = False
                result['error'] = "Incomplete refinement response"

        except Exception as e:
            result['success'] = False
            result['error'] = f"Parsing error: {e}"

        return result

    def compute_self_eval_score(self, self_eval: str) -> float:
        """
        从自评估文本中计算置信度分数

        Args:
            self_eval: 自评估文本

        Returns:
            置信度分数 (0-1)
        """
        if not self_eval:
            return 0.5

        # 定义置信度关键词
        high_confidence_words = ['完全正确', '确信', '准确', '完美', '优秀']
        medium_confidence_words = ['基本正确', '良好', '应该', '可能', '较准确']
        low_confidence_words = ['错误', '不确定', '问题', '需要改进', '不完美']

        self_eval_lower = self_eval.lower()

        # 计算各置信度级别的出现次数
        high_count = sum(1 for word in high_confidence_words if word in self_eval_lower)
        medium_count = sum(1 for word in medium_confidence_words if word in self_eval_lower)
        low_count = sum(1 for word in low_confidence_words if word in self_eval_lower)

        # 计算加权分数
        total_count = high_count + medium_count + low_count
        if total_count == 0:
            return 0.7  # 默认中等偏上

        score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.0) / total_count
        return min(max(score, 0.0), 1.0)

    def validate_generated_sql(self, sql: str) -> Tuple[bool, str]:
        """
        快速验证生成的SQL是否基本有效

        Args:
            sql: SQL字符串

        Returns:
            (是否有效, 错误信息)
        """
        if not sql:
            return False, "Empty SQL"

        # 基本检查
        sql_upper = sql.upper().strip()

        # 检查是否以SELECT开始
        if not sql_upper.startswith('SELECT'):
            return False, "SQL must start with SELECT"

        # 检查括号匹配
        if sql.count('(') != sql.count(')'):
            return False, "Unmatched parentheses"

        # 检查引号匹配
        single_quote_count = sql.count("'")
        double_quote_count = sql.count('"')
        if single_quote_count % 2 != 0 or double_quote_count % 2 != 0:
            return False, "Unmatched quotes"

        return True, ""