"""
SQL证明池管理机制

基于DeepSeek-Math-V2的证明池思想，实现SQL候选查询的管理
包括：候选SQL存储、评分聚合、选择策略、优化组合
"""

import json
import hashlib
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SQLProofPool:
    """SQL证明池管理类"""

    def __init__(self, pool_dir: str = "./outputs/sql_proof_pool"):
        """
        初始化证明池

        Args:
            pool_dir: 池存储目录
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # 证明池配置
        self.max_proofs_per_problem = 20  # 每个问题最大存储的候选数量
        self.n_best_proofs_to_sample = 5   # 选择最佳候选的数量
        self.n_proofs_to_refine = 3       # 用于细化的候选数量

    def add_proof(self, problem_idx: str, proof_data: Dict[str, Any]) -> str:
        """
        添加新的SQL证明到池中

        Args:
            problem_idx: 问题索引
            proof_data: 证明数据：
                {
                    'sql': str,
                    'thinking': str,
                    'self_eval': str,
                    'verifications': List[Dict],  # 验证结果列表
                    'meta_verifications': List[Dict],  # 元验证结果列表
                    'round_number': int,  # 生成的轮次
                    'generation_method': str,  # 生成方法
                    'parent_proof_id': Optional[str]  # 父证明ID（用于细化）
                }

        Returns:
            证明ID
        """
        try:
            # 创建问题目录
            problem_dir = self.pool_dir / problem_idx
            problem_dir.mkdir(exist_ok=True)

            # 生成唯一的证明ID
            proof_id = self._generate_proof_id(proof_data, problem_idx)

            # 计算证明的综合评分
            proof_data = self._calculate_proof_score(proof_data)
            proof_data['proof_id'] = proof_id
            proof_data['timestamp'] = datetime.now().isoformat()

            # 读取现有证明
            pool_file = problem_dir / "proof_pool.jsonl"
            existing_proofs = []
            if pool_file.exists():
                existing_proofs = self._load_proofs_from_file(pool_file)

            # 检查是否已存在相似的证明
            if self._is_duplicate_proof(proof_data, existing_proofs):
                logger.info(f"Duplicate proof detected for problem {problem_idx}, skipping")
                return None

            # 添加新证明
            existing_proofs.append(proof_data)

            # 维护池大小限制
            existing_proofs = self._maintain_pool_size(existing_proofs)

            # 保存到文件
            self._save_proofs_to_file(existing_proofs, pool_file)

            logger.info(f"Added proof {proof_id} to pool for problem {problem_idx}")
            return proof_id

        except Exception as e:
            logger.error(f"Failed to add proof to pool: {e}")
            return None

    def get_best_proofs(self, problem_idx: str, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取最佳证明

        Args:
            problem_idx: 问题索引
            count: 返回的数量（默认使用配置值）

        Returns:
            最佳证明列表，按评分降序排列
        """
        try:
            if count is None:
                count = self.n_best_proofs_to_sample

            pool_file = self.pool_dir / problem_idx / "proof_pool.jsonl"
            if not pool_file.exists():
                return []

            # 加载所有证明
            proofs = self._load_proofs_from_file(pool_file)

            if not proofs:
                return []

            # 按评分排序
            sorted_proofs = sorted(proofs, key=lambda x: x.get('final_score', 0.0), reverse=True)

            return sorted_proofs[:count]

        except Exception as e:
            logger.error(f"Failed to get best proofs: {e}")
            return []

    def get_proof_combinations_for_refinement(self, problem_idx: str) -> List[List[int]]:
        """
        获取用于细化的证明组合

        Args:
            problem_idx: 问题索引

        Returns:
            证明索引组合列表
        """
        try:
            best_proofs = self.get_best_proofs(problem_idx, self.n_best_proofs_to_sample)

            if len(best_proofs) < 2:
                # 候选不足，返回单个最佳证明的索引
                return [[0]] if best_proofs else []

            n_proofs = min(len(best_proofs), self.n_proofs_to_refine)

            # 生成组合策略
            combinations = []

            # 策略1：单独使用前N个最佳证明
            for i in range(n_proofs):
                combinations.append([i])

            # 策略2：使用前2个最佳证明的组合
            if n_proofs >= 2:
                combinations.append([0, 1])

            # 策略3：使用前3个最佳证明的组合
            if n_proofs >= 3:
                combinations.append([0, 1, 2])

            # 策略4：最佳+次佳的交叉组合
            if n_proofs >= 4:
                combinations.append([0, 2])
                combinations.append([1, 3])

            return combinations

        except Exception as e:
            logger.error(f"Failed to get combinations for refinement: {e}")
            return [[0]]  # 默认返回第一个

    def get_pool_stats(self, problem_idx: str) -> Dict[str, Any]:
        """
        获取证明池统计信息

        Args:
            problem_idx: 问题索引

        Returns:
            统计信息字典
        """
        try:
            pool_file = self.pool_dir / problem_idx / "proof_pool.jsonl"
            if not pool_file.exists():
                return {'pool_size': 0}

            proofs = self._load_proofs_from_file(pool_file)

            if not proofs:
                return {'pool_size': 0}

            # 计算统计信息
            scores = [p.get('final_score', 0.0) for p in proofs]
            rounds = [p.get('round_number', 0) for p in proofs]
            methods = [p.get('generation_method', 'unknown') for p in proofs]

            return {
                'pool_size': len(proofs),
                'score_stats': {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'std': self._calculate_std(scores)
                },
                'round_distribution': {r: rounds.count(r) for r in set(rounds)},
                'method_distribution': {m: methods.count(m) for m in set(methods)},
                'latest_timestamp': max(p.get('timestamp', '') for p in proofs)
            }

        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            return {'pool_size': 0, 'error': str(e)}

    def _generate_proof_id(self, proof_data: Dict[str, Any], problem_idx: str) -> str:
        """生成唯一的证明ID"""
        # 基于SQL内容和问题生成哈希
        content = f"{problem_idx}_{proof_data.get('sql', '')}_{proof_data.get('round_number', 0)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _calculate_proof_score(self, proof_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算证明的综合评分

        Args:
            proof_data: 证明数据

        Returns:
            更新后的证明数据（包含评分信息）
        """
        # 获取验证和元验证结果
        verifications = proof_data.get('verifications', [])
        meta_verifications = proof_data.get('meta_verifications', [])

        if not verifications:
            # 没有验证结果，使用自评估分数
            self_eval_score = self._extract_self_eval_score(proof_data.get('self_eval', ''))
            proof_data['mean_score'] = self_eval_score
            proof_data['final_score'] = self_eval_score
            proof_data['score2ratings'] = {self_eval_score: [self_eval_score]}
            return proof_data

        # 计算验证平均分
        verification_scores = [v.get('score', 0.0) for v in verifications if v.get('success', True)]
        mean_verification_score = sum(verification_scores) / len(verification_scores) if verification_scores else 0.0

        # 如果有元验证，使用调整后的分数
        if meta_verifications:
            adjusted_scores = [mv.get('adjusted_score', mv.get('score', 0.0)) for mv in meta_verifications if mv.get('success', True)]
            if adjusted_scores:
                mean_adjusted_score = sum(adjusted_scores) / len(adjusted_scores)
                final_score = (mean_verification_score * 0.4 + mean_adjusted_score * 0.6)
            else:
                final_score = mean_verification_score
        else:
            final_score = mean_verification_score

        # 计算自评估分数
        self_eval_score = self._extract_self_eval_score(proof_data.get('self_eval', ''))

        # 综合评分：验证分数 + 自评估权重
        final_score = final_score * 0.7 + self_eval_score * 0.3

        # 创建评分到评级的映射
        score2ratings = {}
        for v in verifications:
            if v.get('success', True):
                score = round(v.get('score', 0.0), 2)
                score2ratings.setdefault(score, []).append(v.get('score', 0.0))

        proof_data.update({
            'mean_score': mean_verification_score,
            'self_eval_score': self_eval_score,
            'final_score': final_score,
            'score2ratings': score2ratings,
            'verification_count': len(verifications),
            'meta_verification_count': len(meta_verifications)
        })

        return proof_data

    def _extract_self_eval_score(self, self_eval: str) -> float:
        """从自评估文本中提取分数"""
        if not self_eval:
            return 0.5

        # 简单的关键词匹配
        positive_words = ['正确', '良好', '完整', '合适', '准确']
        negative_words = ['错误', '问题', '不完美', '需要改进', '不确定']

        self_eval_lower = self_eval.lower()
        pos_count = sum(1 for word in positive_words if word in self_eval_lower)
        neg_count = sum(1 for word in negative_words if word in self_eval_lower)

        total_words = pos_count + neg_count
        if total_words == 0:
            return 0.7  # 默认中等偏上

        score = (pos_count * 1.0 + neg_count * 0.0) / total_words
        return max(0.0, min(1.0, score))

    def _is_duplicate_proof(self, new_proof: Dict[str, Any], existing_proofs: List[Dict[str, Any]]) -> bool:
        """检查是否为重复证明"""
        new_sql = new_proof.get('sql', '').strip()

        for existing_proof in existing_proofs:
            existing_sql = existing_proof.get('sql', '').strip()

            # 简单的SQL完全匹配检查
            if new_sql == existing_sql:
                return True

            # 可以添加更复杂的相似性检查
            # 例如：忽略大小写、空白字符的差异
            if new_sql.lower().replace(' ', '') == existing_sql.lower().replace(' ', ''):
                return True

        return False

    def _maintain_pool_size(self, proofs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """维护池大小，保留最佳证明"""
        if len(proofs) <= self.max_proofs_per_problem:
            return proofs

        # 按评分排序，保留最佳证明
        sorted_proofs = sorted(proofs, key=lambda x: x.get('final_score', 0.0), reverse=True)
        return sorted_proofs[:self.max_proofs_per_problem]

    def _load_proofs_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """从文件加载证明"""
        proofs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        proofs.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"Failed to load proofs from {file_path}: {e}")

        return proofs

    def _save_proofs_to_file(self, proofs: List[Dict[str, Any]], file_path: Path):
        """保存证明到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for proof in proofs:
                    f.write(json.dumps(proof, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save proofs to {file_path}: {e}")

    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def clear_pool(self, problem_idx: str = None):
        """清空证明池"""
        try:
            if problem_idx:
                # 清空特定问题的池
                problem_dir = self.pool_dir / problem_idx
                if problem_dir.exists():
                    for file in problem_dir.glob("*.jsonl"):
                        file.unlink()
                    logger.info(f"Cleared proof pool for problem {problem_idx}")
            else:
                # 清空所有池
                for problem_dir in self.pool_dir.iterdir():
                    if problem_dir.is_dir():
                        for file in problem_dir.glob("*.jsonl"):
                            file.unlink()
                logger.info("Cleared all proof pools")
        except Exception as e:
            logger.error(f"Failed to clear proof pool: {e}")