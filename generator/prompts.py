"""
提示词模板库
所有的提示词都集中在这里管理
"""

class PromptTemplates:
    """NL2SQL提示词模板集合"""

    # ========== 基础提示词 ==========
    BASE_SYSTEM_PROMPT = """
你是一个具备强大思维链（thinking mode）能力的 NL2SQL 专家模型，基于 Qwen3 架构并参考 DeepSeek-Math-V2 的自验证训练方法工作。

你的任务：
1. 在 <think>...</think> 段中给出完整、结构化、可自我检查的推理过程；
2. 在 <answer>...</answer> 段中只输出最终 SQL 语句（可多行），不得包含任何自然语言解释。

在 <think> 中你需要：
- 梳理问题信息：时间范围、地域、维度、指标、筛选条件、分组和聚合逻辑；
- 结合 schema 选择正确的表和字段；
- 写出候选 SQL 的草稿，并自检可能的问题（如缺少时间范围、JOIN 条件、GROUP BY 不完整、括号不匹配等）；
- 如果发现问题，在思考中说明并修正。
"""


    # ========== SQL生成提示词 ==========
    SQL_GENERATION_PROMPT = """
[系统准则]
{system_prompt}

[业务背景]
{business_knowledge}

[数据表结构]
{schema}

[示例]
{few_shot_examples}

[当前问题]
{question}

请严格按照以下格式回答（不得省略标签）：

<think>
在这里详细写出你的思考过程，包含：
- 对问题的结构化理解（时间、地域、指标、度量）；
- 如何选择相关表和字段；
- 如何设计 WHERE、JOIN、GROUP BY、HAVING 等；
- 对候选 SQL 的自我检查和修正过程。
</think>
<answer>
在这里只输出最终 SQL 语句，不包含任何自然语言说明。
</answer>
"""


    # ========== 自我评估提示词 ==========
    SELF_ASSESSMENT_PROMPT = """现在请对你生成的SQL进行自我评估。

## 评估标准
1. **简单SQL (0分)**：语法错误、逻辑错误或完全偏离问题
2. **部分正确 (0.5分)**：逻辑合理但有小错误或缺少细节
3. **完全正确 (1分)**：完全符合需求，逻辑清晰

## 评估内容
请检查以下方面：
- WHERE条件是否完整？
- 时间范围是否正确指定？
- JOIN条件是否完备？
- GROUP BY是否与聚合函数配套？
- 字段名称是否正确？
- 表别名是否一致？

## 输出格式
请在 <evaluation> 和 </evaluation> 标签中给出评估：

<evaluation>
问题分析：[对SQL可能存在的问题的分析]
最终评分：\\boxed{0/0.5/1}
</evaluation>
"""

    # ========== 任务类型分类提示词 ==========
    TASK_TYPE_CLASSIFICATION_PROMPT = """根据生成的SQL和问题特征，判断其任务类型：

## 分类标准
- **SQL**：简单思考直接输出SQL
- **多步推理**：多步骤思考，输出可能带有CTE等的复杂SQL
- **反思**：将输入的错误SQL更正
- **歧义澄清**：用户问题包含歧义点，触发模型思考
- **维度拒识**：用户问题包含查询不支持的维度时模型拒绝回答
- **维度退化**：维表退化到事实表时仍支持查询
- **指标拒识**：用户问题包含查询不支持的指标时模型拒绝回答
- **追问**：用户问题不满足查询的必备要求

## 判断依据
分析以下因素：
- JOIN的数量和复杂度
- 是否使用了WITH (CTE)
- 是否有UNION操作
- 子查询的数量
- 是否为错误SQL修正
- 是否包含拒识或澄清逻辑
- 是否为追问场景

## 输出
任务类型：[SQL/多步推理/反思/歧义澄清/维度拒识/维度退化/指标拒识/追问]
"""

    # ========== Few-shot示例 ==========
    @staticmethod
    def get_few_shot_examples():
        """获取示例"""
        return """
### 示例1：简单查询
**问题**：今年的总线索量是多少？

<think>
问题要求查询今年的线索总量。
- 时间范围：今年，year_id = YEAR(CURDATE())
- 指标：净新增线索量，COUNT(DISTINCT customer_account_id)
- 表：dwd_ai_dc_sale_retail_leads_net_df
- 过滤条件：is_net_leads = 1
</think>

<answer>
SELECT
    COUNT(DISTINCT customer_account_id) AS leads_cnt
FROM dwd_ai_dc_sale_retail_leads_net_df t1
LEFT JOIN dim_ai_dc_day t2 ON t1.day_id = t2.day_id
WHERE
    is_net_leads = '1'
    AND t2.year_id = YEAR(CURDATE())
</answer>

### 示例2：复杂查询
**问题**：按月份查看去年与今年同期的线索量增长

<think>
这是一个时间对比分析问题。
- 需要关联两个时间段的数据
- 使用CTE分别计算去年和今年的数据
- 然后进行对比计算增长率
- 分组维度：月份
</think>

<answer>
WITH last_year AS (
    SELECT
        t2.month_id,
        COUNT(DISTINCT t1.customer_account_id) AS leads_cnt
    FROM dwd_ai_dc_sale_retail_leads_net_df t1
    LEFT JOIN dim_ai_dc_day t2 ON t1.day_id = t2.day_id
    WHERE t2.year_id = YEAR(CURDATE()) - 1
    GROUP BY t2.month_id
),
this_year AS (
    SELECT
        t2.month_id,
        COUNT(DISTINCT t1.customer_account_id) AS leads_cnt
    FROM dwd_ai_dc_sale_retail_leads_net_df t1
    LEFT JOIN dim_ai_dc_day t2 ON t1.day_id = t2.day_id
    WHERE t2.year_id = YEAR(CURDATE())
    GROUP BY t2.month_id
)
SELECT
    t.month_id,
    l.leads_cnt AS last_year_leads,
    t.leads_cnt AS this_year_leads,
    ROUND((t.leads_cnt - l.leads_cnt) * 100.0 / l.leads_cnt, 2) AS growth_rate
FROM this_year t
LEFT JOIN last_year l ON t.month_id = l.month_id
ORDER BY t.month_id
</answer>
"""

    # ========== 业务知识模板 ==========
    @staticmethod
    def get_business_knowledge():
        """获取业务知识"""
        return """
### 时间处理规则
- 默认时间为当年
- 最近N天：day_id BETWEEN DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL N DAY), '%Y%m%d') AND DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y%m%d')
- 本月至今：month_id = DATE_FORMAT(CURDATE(), '%Y%m') AND day_id < REPLACE(CURDATE(), '-', '')

### 维度分组默认规则
- 按渠道分组：默认按一级渠道 (first_leads_channel_tag_name)
- 按地域分组：默认按城市 (city_name)
- 按车分组：默认按车系 (series_name)

### 指标计算
- 净新增线索量 = COUNT(DISTINCT customer_account_id) WHERE is_net_leads = 1
- 线索目标值 = SUM(target_cnt) WHERE module = '线索'
"""

    @staticmethod
    def get_schema_context():
        """获取数据表结构"""
        return """
### 主要维度表
1. **dim_ai_dc_day** (时间维度表)
   - day_id: 日期 (YYYYMMDD)
   - month_id: 月份 (YYYYMM)
   - quarter_id: 季度 (YYYYQ)
   - year_id: 年份 (YYYY)

2. **dim_ai_dc_area_df** (地域维度表)
   - city_id: 城市编码
   - city_name: 城市名称
   - province_name: 省份名称
   - city_level: 城市等级

3. **dim_ai_dc_brand_series_df** (品牌车系表)
   - series_id: 车系ID
   - series_name: 车系名称
   - brand_name: 品牌名称

### 主要事实表
1. **dwd_ai_dc_sale_retail_leads_net_df** (线索明细表)
   - day_id: 日期
   - customer_account_id: 用户ID
   - city_id: 城市ID
   - series_id: 车系ID
   - is_net_leads: 是否净新增线索 (1=是)
   - first_leads_channel_tag_name: 一级渠道名称
   - second_leads_channel_tag_name: 二级渠道名称
   - third_leads_channel_tag_name: 三级渠道名称
   - fourth_leads_channel_tag_name: 四级渠道名称
   - dept_id: 部门ID

2. **dwd_ai_dc_sale_target_df** (销售目标表)
   - day_id: 日期
   - target_cnt: 目标值
   - module: 数据域 (例如: '线索')
   - first_leads_channel_tag_name: 一级渠道名称
   - target_level_id: 目标等级 (0=无指定省市, 20=有指定省市)
"""
