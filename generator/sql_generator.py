"""
SQL生成器
调用模型生成SQL查询并自动分类复杂度
"""

import logging
import re
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from classifiers.complexity_classifier import ComplexityClassifier
from classifiers.meta_classifier import MetaClassifier
from generator.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class SQLGenerator:
    """
    SQL生成器

    功能：
    1. 加载Qwen模型
    2. 生成SQL及推理过程
    3. 自动分类复杂度
    4. 进行自我评估
    """

    def __init__(self,
                 model_name: str,
                 device: str = "cuda",
                 torch_dtype=torch.bfloat16,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 top_k: int = 50,
                 max_new_tokens: int = 1024):
        """
        初始化SQL生成器

        Args:
            model_name: 模型名称 (e.g., "Qwen/Qwen2.5-14B-Instruct")
            device: 设备 (cuda或cpu)
            torch_dtype: torch数据类型
            temperature: 生成温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            max_new_tokens: 最大生成令牌数
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        logger.info(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        logger.info(f"模型已加载到: {device}")

        # 初始化复杂度分类器和元分类器
        self.complexity_classifier = ComplexityClassifier()
        self.meta_classifier = MetaClassifier()

    def generate(self,
                 question: str,
                 schema: str,
                 business_knowledge: Optional[str] = None,
                 few_shot_examples: Optional[str] = None) -> Dict:
        """
        生成SQL查询

        Args:
            question: 用户问题
            schema: 数据表结构
            business_knowledge: 业务知识背景
            few_shot_examples: 少样本示例

        Returns:
            生成结果字典
        """
        # 使用默认的业务知识和示例
        if business_knowledge is None:
            business_knowledge = PromptTemplates.get_business_knowledge()
        if few_shot_examples is None:
            few_shot_examples = PromptTemplates.get_few_shot_examples()

        # 构造提示词
        system_prompt = PromptTemplates.BASE_SYSTEM_PROMPT
        user_prompt = PromptTemplates.SQL_GENERATION_PROMPT.format(
            system_prompt=system_prompt,
            schema=schema,
            business_knowledge=business_knowledge,
            few_shot_examples=few_shot_examples,
            question=question
        )

        # 调用模型
        try:
            response = self._call_model(user_prompt)
        except Exception as e:
            logger.error(f"模型推理出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'question': question
            }

        # 解析响应
        thinking, sql = self._extract_think_answer(response)

        # 分类复杂度
        classification_result = self.complexity_classifier.classify(sql)

        # 验证分类结果
        is_reasonable, confidence = self.meta_classifier.verify_classification(
            sql,
            classification_result.complexity,
            thinking
        )

        # 生成自我评估
        self_assessment = self._generate_self_assessment(sql, classification_result)

        result = {
            'success': True,
            'question': question,
            'thinking': thinking,
            'sql': sql,
            'complexity_type': classification_result.complexity,
            'raw_response': response,
            'classification': {
                'type': classification_result.complexity,
                'confidence': classification_result.confidence,
                'severity_score': classification_result.severity_score,
                'detected_issues': [
                    {
                        'type': issue.issue_type,
                        'severity': issue.severity,
                        'description': issue.description
                    }
                    for issue in classification_result.issue_details
                ]
            },
            'verification': {
                'is_reasonable': is_reasonable,
                'confidence': confidence,
                'thinking_quality': self.meta_classifier._check_thinking_quality(thinking),
                'type_match': self.meta_classifier._check_type_match(sql, classification_result.complexity),
                'sql_validity': self.meta_classifier._check_sql_validity(sql)
            },
            'self_assessment': self_assessment
        }

        return result

    def _call_model(self, prompt: str) -> str:
        """
        调用模型生成响应

        Args:
            prompt: 输入提示词

        Returns:
            模型生成的响应
        """
        # 应用聊天模板
        messages = [
            {"role": "user", "content": prompt}
        ]

        # 使用apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response

    @staticmethod
    def _extract_think_answer(response: str) -> Tuple[str, str]:
        """
        从响应中提取<think>和<answer>部分

        Args:
            response: 模型响应

        Returns:
            (thinking, sql)
        """
        # 提取<think>部分
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""

        # 提取<answer>部分
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        sql = answer_match.group(1).strip() if answer_match else ""

        return thinking, sql

    @staticmethod
    def _generate_self_assessment(sql: str, classification_result) -> Dict:
        """
        生成自我评估

        Args:
            sql: SQL查询
            classification_result: 复杂度分类结果

        Returns:
            自我评估字典
        """
        assessment = {
            'has_where': 'WHERE' in sql.upper(),
            'has_time_range': any(
                t in sql.upper()
                for t in ['DAY_ID', 'MONTH_ID', 'YEAR_ID', 'QUARTER_ID']
            ),
            'has_join': 'JOIN' in sql.upper(),
            'has_group_by': 'GROUP BY' in sql.upper(),
            'has_agg_function': any(
                agg in sql.upper()
                for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']
            ),
            'brackets_balanced': sql.count('(') == sql.count(')'),
            'critical_issues': len([
                issue for issue in classification_result.issue_details
                if issue.severity >= 0.5
            ]),
            'overall_quality': 'good' if classification_result.severity_score < 0.3
                              else ('fair' if classification_result.severity_score < 0.6
                                   else 'poor'),
            'recommendation': (
                "SQL质量良好，可用于训练" if classification_result.severity_score < 0.3
                else "SQL有一些问题，需要人工审查" if classification_result.severity_score < 0.6
                else "SQL存在严重问题，建议重新生成"
            )
        }
        return assessment

    def batch_generate(self,
                       questions: list,
                       schemas: list,
                       batch_size: int = 4) -> list:
        """
        批量生成SQL

        Args:
            questions: 问题列表
            schemas: 数据表结构列表
            batch_size: 批处理大小

        Returns:
            结果列表
        """
        results = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_schemas = schemas[i:i+batch_size]

            for question, schema in zip(batch_questions, batch_schemas):
                result = self.generate(question, schema)
                results.append(result)
                logger.info(
                    f"生成进度: {i+1}/{len(questions)} - "
                    f"复杂度: {result.get('complexity_type', 'N/A')} - "
                    f"成功: {result.get('success', False)}"
                )

        return results

    def free_memory(self):
        """释放显存"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        logger.info("已释放显存")
