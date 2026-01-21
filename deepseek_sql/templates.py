"""
SQL领域的提示模板

基于DeepSeek-Math-V2的模板设计，适配到SQL生成任务
包含：生成模板、验证模板、元验证模板
"""

from typing import Dict, Any


class SQLTemplates:
    """SQL提示模板类"""

    def __init__(self):
        """初始化SQL模板"""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """加载所有模板"""
        return {
            # ==================== SQL生成模板 ====================
            "sql_generation": """你是一个资深的数据分析师和SQL专家。你的任务是根据用户的自然语言问题，准确地生成对应的SQL查询。

你需要完成以下步骤：

1. 【理解问题】仔细分析用户的问题，识别出：
   - 需要查询的指标（如线索量、订单数等）
   - 时间范围约束
   - 维度分组要求
   - 其他过滤条件

2. 【构建SQL】基于数据库Schema和业务知识，生成规范的SQL查询：
   - 选择正确的表和字段
   - 添加合适的JOIN条件
   - 包含必要的WHERE条件（特别是时间范围）
   - 使用正确的聚合函数

3. 【自我评估】生成SQL后，进行严格的自检：
   - 检查语法是否正确
   - 验证逻辑是否符合问题要求
   - 确认所有必要的条件都已包含
   - 评估可能存在的问题

你的回答必须严格按照以下格式：

<thinking>
[你的详细推理过程，包括问题分析、Schema理解、SQL构建逻辑等]
</thinking>

<sql>
[你生成的完整SQL查询]
</sql>

<self_eval>
[你的自我评估结果，包括：
- SQL的语法正确性
- 逻辑完整性
- 潜在问题分析
- 置信度评估]
</self_eval>

请基于以下信息回答：

问题：{query}

数据库Schema：
{schema}

业务知识：
{knowledge}

示例：
{examples}

严格按照上述格式回答，不要遗漏任何部分。""",

            # ==================== SQL验证模板 ====================
            "sql_verification": """你是一个SQL验证专家。你的任务是对给定的SQL查询进行严格的验证和评分。

你需要从以下几个维度评估SQL的质量：

1. **语法正确性** (30%)
   - SQL语法是否符合规范
   - 关键字使用是否正确
   - 括号、引号是否匹配

2. **逻辑正确性** (40%)
   - 是否正确理解了原问题
   - JOIN条件是否完整
   - WHERE条件是否准确
   - 聚合函数使用是否得当

3. **业务一致性** (20%)
   - 是否符合业务规则
   - 时间范围处理是否正确
   - 指标计算是否符合定义

4. **查询效率** (10%)
   - 是否存在明显的性能问题
   - 表连接方式是否合理

评分标准：
- 1.0分：完全正确，无需修改
- 0.5分：基本正确但有细节问题
- 0.0分：存在严重错误

请按照以下格式回答：

<analysis>
[你的详细分析过程，解释为什么给出这个评分]
</analysis>

<score>
[0.0, 0.5, 或 1.0]
</score>

<issues>
[列出发现的所有问题，如果没有问题则写"无"]
</issues>

请验证以下SQL：

原始问题：{query}
生成的SQL：{sql}
数据库Schema：{schema}
业务知识：{knowledge}

严格按照上述格式回答。""",

            # ==================== SQL元验证模板 ====================
            "sql_meta_verification": """你是一个质量控制专家。你的任务是评估SQL验证器的评分是否合理。

你需要考虑：

1. **验证逻辑的合理性**
   - 验证器是否正确识别了SQL的问题
   - 问题的严重程度判断是否准确
   - 评分是否符合标准

2. **一致性检查**
   - 对于同样的错误类型，评分是否一致
   - 对于良好的SQL，是否给予了合理的分数

3. **评估完整性**
   - 是否遗漏了重要的问题
   - 是否对无问题的SQL给予了公正评价

如果验证器的评估不够准确，你需要给出更合理的评分和解释。

请按照以下格式回答：

<meta_analysis>
[你对验证器评分的分析和评估]
</meta_analysis>

<adjusted_score>
[如果你认为原来的分数合理，就保持不变；如果不合理，给出调整后的分数：0.0, 0.5, 或 1.0]
</adjusted_score>

<confidence>
[你对这个最终评分的置信度：高/中/低]
</confidence>

信息汇总：
- 原始问题：{query}
- 生成的SQL：{sql}
- 验证器评分：{verifier_score}
- 验证器分析：{verifier_analysis}
- 发现的问题：{issues}

请给出你的元评估结果。""",

            # ==================== SQL细化模板 ====================
            "sql_refinement": """你是一个SQL优化专家。基于多轮验证反馈，你需要生成一个改进的SQL查询。

收到的反馈包括：
- 多个验证器的评分
- 发现的具体问题
- 改进建议

你的任务：
1. 综合分析所有反馈
2. 理解主要问题所在
3. 生成改进的SQL
4. 更新你的推理过程

请按照以下格式回答：

<refined_thinking>
[基于反馈的改进思考过程]
</refined_thinking>

<refined_sql>
[改进后的SQL查询]
</refined_sql>

<refined_self_eval>
[对改进SQL的自我评估]
</refined_self_eval>

反馈信息：
- 原始问题：{query}
- 原始SQL：{original_sql}
- 验证反馈：{verification_feedback}
- 平均评分：{average_score}

请生成改进的SQL。"""
        }

    def get_template(self, template_name: str) -> str:
        """获取指定模板"""
        return self.templates.get(template_name, "")

    def format_template(self, template_name: str, **kwargs) -> str:
        """格式化模板"""
        template = self.get_template(template_name)
        return template.format(**kwargs)

    def get_generation_template(self, query: str, schema: str, knowledge: str, examples: str = "") -> str:
        """获取SQL生成模板"""
        return self.format_template(
            "sql_generation",
            query=query,
            schema=schema,
            knowledge=knowledge,
            examples=examples
        )

    def get_verification_template(self, query: str, sql: str, schema: str, knowledge: str) -> str:
        """获取SQL验证模板"""
        return self.format_template(
            "sql_verification",
            query=query,
            sql=sql,
            schema=schema,
            knowledge=knowledge
        )

    def get_meta_verification_template(self, query: str, sql: str, verifier_score: str,
                                     verifier_analysis: str, issues: str, schema: str,
                                     knowledge: str) -> str:
        """获取SQL元验证模板"""
        return self.format_template(
            "sql_meta_verification",
            query=query,
            sql=sql,
            verifier_score=verifier_score,
            verifier_analysis=verifier_analysis,
            issues=issues,
            schema=schema,
            knowledge=knowledge
        )

    def get_refinement_template(self, query: str, original_sql: str, verification_feedback: str,
                              average_score: float) -> str:
        """获取SQL细化模板"""
        return self.format_template(
            "sql_refinement",
            query=query,
            original_sql=original_sql,
            verification_feedback=verification_feedback,
            average_score=average_score
        )