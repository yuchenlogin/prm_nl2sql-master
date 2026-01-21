"""
独立的VLM验证器

基于SFT微调后的Qwen3-1.7B模型，实现真正的SQL验证功能
这是对原始DeepSeek-Math-V2 VLM机制的正确实现
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class VLMVerifier:
    """
    独立的VLM验证器

    与生成模型分离，提供客观的推理步骤验证
    """

    def __init__(self, model_path: str = "/lpai/models/Qwen3-1.7B-SFT"):
        """
        初始化VLM验证器

        Args:
            model_path: SFT微调后的模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        logger.info(f"Loading VLM verifier from {model_path}")

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"VLM verifier loaded successfully on {self.device}")

    def verify_reasoning_steps(
        self,
        problem: str,
        reasoning_steps: List[str],
        schema: str = "",
        knowledge: str = ""
    ) -> Dict[str, Any]:
        """
        验证推理步骤的正确性

        Args:
            problem: 原始问题
            reasoning_steps: 推理步骤列表
            schema: 数据库Schema（可选）
            knowledge: 业务知识（可选）

        Returns:
            验证结果字典
        """
        try:
            # 构建验证提示
            prompt = self._build_verification_prompt(
                problem, reasoning_steps, schema, knowledge
            )

            # 生成验证结果
            with torch.no_grad():
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                inputs = inputs.to(self.model.device)

                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.3,  # VLM采用较低温度，增强确定性
                    top_p=0.9,
                    top_k=50,
                    do_sample=False,  # 确定性输出
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # 解析验证结果
            parsed_result = self._parse_verification_response(response)

            return {
                'success': True,
                'problem': problem,
                'reasoning_steps': reasoning_steps,
                'verification': parsed_result,
                'full_response': response,
                'verifier_type': 'VLM'
            }

        except Exception as e:
            logger.error(f"VLM verification failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'verifier_type': 'VLM'
            }

    def _build_verification_prompt(
        self,
        problem: str,
        reasoning_steps: List[str],
        schema: str,
        knowledge: str
    ) -> str:
        """
        构建验证提示

        VLM的任务是评估推理步骤，而不是生成SQL
        """
        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(reasoning_steps)])

        prompt = f"""你是一个专业的SQL推理验证专家。你的任务是评估给定的推理步骤是否正确、完整。

【问题】
{problem}

【推理步骤】
{steps_text}
"""

        if schema:
            prompt += f"""
【数据库Schema】
{schema}
"""

        if knowledge:
            prompt += f"""
【业务知识】
{knowledge}
"""

        prompt += """
请从以下维度评估推理步骤：

1. **步骤逻辑性**：每个步骤是否合乎逻辑
2. **条件完整性**：是否遗漏了必要的过滤条件
3. **时间处理**：如果有时间约束，是否正确处理
4. **语法正确性**：构建SQL的语法是否正确
5. **潜在问题**：是否存在逻辑漏洞或错误

请按以下格式输出：

<analysis>
[你的详细分析]
</analysis>

<issues>
如果发现问题，列出具体问题。如果没有，写"未发现问题"
</issues>

<confidence>
[你的置信度打分，例如：0.8]
</confidence>

<score>
[整体评分，0-1之间]
</score>
"""

        return prompt

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """
        解析VLM的验证响应

        Args:
            response: VLM输出的验证文本

        Returns:
            解析后的验证结果
        """
        import re

        result = {
            'analysis': "",
            'issues': "",
            'confidence': 0.5,
            'score': 0.5
        }

        try:
            # 提取analysis
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL)
            if analysis_match:
                result['analysis'] = analysis_match.group(1).strip()

            # 提取issues
            issues_match = re.search(r'<issues>(.*?)</issues>', response, re.DOTALL)
            if issues_match:
                result['issues'] = issues_match.group(1).strip()

            # 提取confidence
            conf_match = re.search(r'<confidence>(.*?)</confidence>', response, re.DOTALL)
            if conf_match:
                conf_text = conf_match.group(1).strip()
                # 尝试提取数字
                conf_num = re.search(r'\d+(\.\d+)?', conf_text)
                if conf_num:
                    result['confidence'] = float(conf_num.group())
                else:
                    # 根据关键词判断
                    if any(kw in conf_text.lower() for kw in ['高', '肯定', '确信']):
                        result['confidence'] = 0.8
                    elif any(kw in conf_text.lower() for kw in ['低', '不确定', '可能']):
                        result['confidence'] = 0.3
                    else:
                        result['confidence'] = 0.5

            # 提取score
            score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
            if score_match:
                score_text = score_match.group(1).strip()
                score_num = re.search(r'(\d+(\.\d+)?)', score_text)
                if score_num:
                    result['score'] = float(score_num.group())
                    # 确保在0-1之间
                    result['score'] = max(0.0, min(1.0, result['score']))

            # 基于issues调整scores
            if "未发现问题" in result.get('issues', ''):
                # 没有发现问题，给高分
                result['score'] = max(0.8, result['score'])
            else:
                # 有问题，降低分数
                result['score'] = result['score'] * 0.7

        except Exception as e:
            logger.warning(f"Error parsing VLM response: {e}")
            result = {
                'analysis': response[:500],
                'issues': "解析错误",
                'confidence': 0.5,
                'score': 0.5
            }

        return result

    def verify_sql(
        self,
        query: str,
        sql: str,
        schema: str = "",
        knowledge: str = ""
    ) -> Dict[str, Any]:
        """
        验证完整的SQL（包括推理和SQL本身）

        Args:
            query: 用户问题
            sql: 生成的SQL
            schema: 数据库Schema
            knowledge: 业务知识

        Returns:
            验证结果字典
        """
        # 从SQL提取推理步骤（如果有的话）
        reasoning_steps = self._extract_reasoning_from_sql(sql)

        # 验证推理步骤
        verification = self.verify_reasoning_steps(
            query, reasoning_steps, schema, knowledge
        )

        # 直接验证SQL语法和结构
        sql_verification = self._verify_sql_structure(sql, schema)

        # 综合评分
        final_score = (
            verification.get('verification', {}).get('score', 0.5) * 0.7 +
            sql_verification.get('score', 0.5) * 0.3
        )

        return {
            'success': verification.get('success', False),
            'score': final_score,
            'vlm_verification': verification,
            'structural_verification': sql_verification,
            'reasoning_steps': reasoning_steps,
            'verifier_type': 'VLM'
        }

    def _extract_reasoning_from_sql(self, sql: str) -> List[str]:
        """
        从SQL中提取推理步骤

        如果SQL包含注释，尝试提取逻辑
        """
        steps = []

        # 提取注释
        import re
        comments = re.findall(r'--\s*(.+)', sql)
        if comments:
            steps.extend(comments)

        # 如果没有注释，从SQL结构推断
        if not steps:
            # SELECT部分
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                steps.append(f"选择字段：{select_match.group(1).strip()}")

            # FROM部分（表）
            from_match = re.search(r'FROM\s+(\S+)', sql, re.IGNORECASE)
            if from_match:
                steps.append(f"查询表：{from_match.group(1)}")

            # WHERE部分
            where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
            if where_match:
                steps.append(f"过滤条件：{where_match.group(1).strip()}")

            # GROUP BY部分
            group_match = re.search(r'GROUP BY\s+(.+?)(?:\s+ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
            if group_match:
                steps.append(f"分组依据：{group_match.group(1).strip()}")

        return steps

    def _verify_sql_structure(self, sql: str, schema: str) -> Dict[str, Any]:
        """
        基于规则的SQL结构验证

        这是确定性的验证，作为VLM的补充
        """
        score = 1.0
        issues = []

        sql_upper = sql.upper().strip()

        # 1. 检查是否以SELECT开始
        if not sql_upper.startswith('SELECT'):
            score -= 0.5
            issues.append("SQL必须以SELECT开始")

        # 2. 检查括号匹配
        if sql.count('(') != sql.count(')'):
            score -= 0.3
            issues.append("括号不匹配")

        # 3. 检查引号匹配
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            score -= 0.2
            issues.append("单引号不匹配")

        # 4. 检查JOIN语法
        if 'JOIN' in sql_upper and 'ON' not in sql_upper:
            score -= 0.3
            issues.append("JOIN需要ON子句")

        # 5. 检查分号结尾
        if not sql.strip().endswith(';'):
            issues.append("建议SQL以分号结尾")
            # 这不是严重错误，不减分

        # 6. 检查空SQL
        if len(sql.strip()) < 10:
            score = 0.0
            issues.append("SQL内容为空")

        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))

        return {
            'score': score,
            'issues': issues,
            'pass_check': score > 0.6
        }

    def evaluate_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量验证（用于训练时的高效验证）

        Args:
            batch: 包含多个验证任务的列表

        Returns:
            验证结果列表
        """
        results = []
        for item in batch:
            result = self.verify_sql(
                query=item.get('query', ''),
                sql=item.get('sql', ''),
                schema=item.get('schema', ''),
                knowledge=item.get('knowledge', '')
            )
            results.append(result)
        return results
