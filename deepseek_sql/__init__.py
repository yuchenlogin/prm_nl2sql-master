"""
DeepSeek SQL适配模块

基于DeepSeek-Math-V2的自验证机制，适配到NL2SQL任务
实现三层验证架构：SQL生成→验证→元验证
"""

from .main import DeepSeekNL2SQL
from .templates import SQLTemplates
from .generators import SQLGenerator
from .verifiers import SQLVerifier
from .meta_verifiers import SQLMetaVerifier
from .proof_pool import SQLProofPool
from .reward_calculator import SQLProcessRewardCalculator

__all__ = [
    "DeepSeekNL2SQL",
    "SQLTemplates",
    "SQLGenerator",
    "SQLVerifier",
    "SQLMetaVerifier",
    "SQLProofPool",
    "SQLProcessRewardCalculator"
]

__version__ = "1.0.0"