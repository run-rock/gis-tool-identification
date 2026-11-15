#!/usr/bin/env python3
"""
统一的评估器模块
基于字符串匹配进行评估
支持并行评估
"""

import json
import math
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class RetrievalEvaluator:
    """检索评估器（基于字符串匹配）"""

    def __init__(self, vector_db, k_values: List[int] = [1, 3, 5], max_workers: int = 4):
        """
        初始化评估器

        Args:
            vector_db: FAISSVectorDatabase实例
            k_values: 评估的k值列表
            max_workers: 并行评估的最大线程数
        """
        self.vector_db = vector_db
        self.k_values = sorted(k_values)
        self.max_k = max(k_values)
        self.max_workers = max_workers
        self.model_lock = threading.Lock()  # 用于线程安全访问向量库

        print(f"✅ 检索评估器初始化完成（基于字符串匹配）")
        print(f"   评估指标: Recall@{k_values}, NDCG@{k_values}")
        print(f"   向量库大小: {len(self.vector_db.texts)} 条文档")
        print(f"   并行线程数: {max_workers}")

    def find_answer_indices_by_string(self, correct_answers: List[str]) -> List[int]:
        """
        根据字符串匹配查找答案在向量库中的索引

        Args:
            correct_answers: 正确答案字符串列表

        Returns:
            找到的索引列表
        """
        found_indices = []

        for answer in correct_answers:
            # 在向量库中查找完全匹配的文本
            try:
                idx = self.vector_db.texts.index(answer)
                found_indices.append(idx)
            except ValueError:
                # 未找到匹配
                pass

        return found_indices

    def calculate_recall_at_k(self, retrieved_indices: List[int],
                               relevant_indices: List[int], k: int) -> float:
        """
        计算 Recall@k

        Args:
            retrieved_indices: 检索到的文档索引列表（按相关性排序）
            relevant_indices: 相关文档的索引列表
            k: top-k

        Returns:
            Recall@k 分数
        """
        if not relevant_indices:
            return 0.0

        # 取前k个检索结果
        top_k_retrieved = set(retrieved_indices[:k])
        relevant_set = set(relevant_indices)

        # 计算召回率：命中至少一个正确答案即为1
        if top_k_retrieved & relevant_set:
            return 1.0
        else:
            return 0.0

    def calculate_ndcg_at_k(self, retrieved_indices: List[int],
                            relevant_indices: List[int], k: int,
                            relevance_scores: Optional[List[float]] = None) -> float:
        """
        计算 NDCG@k (Normalized Discounted Cumulative Gain)

        Args:
            retrieved_indices: 检索到的文档索引列表（按相关性排序）
            relevant_indices: 相关文档的索引列表
            k: top-k
            relevance_scores: 相关性分数列表（如果为None，则相关文档分数为1）

        Returns:
            NDCG@k 分数
        """
        if not relevant_indices:
            return 0.0

        # 默认相关性分数：相关文档为1，不相关为0
        if relevance_scores is None:
            relevance_map = {idx: 1.0 for idx in relevant_indices}
        else:
            relevance_map = {relevant_indices[i]: score for i, score in enumerate(relevance_scores)}

        # 计算DCG@k
        dcg = 0.0
        for i, doc_idx in enumerate(retrieved_indices[:k]):
            if doc_idx in relevance_map:
                rel_score = relevance_map[doc_idx]
                # DCG公式: rel_i / log2(i + 2)
                dcg += rel_score / math.log2(i + 2)

        # 计算IDCG@k (Ideal DCG)
        ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / math.log2(i + 2)

        # 计算NDCG
        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

    def evaluate_single_query(self, query: str,
                               correct_answers: List[str],
                               relevance_scores: Optional[List[float]] = None) -> Dict:
        """
        评估单个查询

        Args:
            query: 查询文本
            correct_answers: 正确答案文本列表
            relevance_scores: 相关性分数列表（可选）

        Returns:
            评估结果字典
        """
        # 根据字符串匹配查找正确答案在向量库中的索引
        correct_answer_indices = self.find_answer_indices_by_string(correct_answers)

        if not correct_answer_indices:
            # 所有正确答案都未在向量库中找到
            metrics = {}
            for k in self.k_values:
                metrics[f"recall@{k}"] = 0.0
                metrics[f"ndcg@{k}"] = 0.0

            return {
                "query": query,
                "correct_answers": correct_answers,
                "correct_answer_indices": [],
                "found_answers_count": 0,
                "total_answers_count": len(correct_answers),
                "retrieved_indices": [],
                "metrics": metrics,
                "search_results": []
            }

        # 执行检索
        search_results = self.vector_db.search(query, k=self.max_k)
        retrieved_indices = [result["index"] for result in search_results["results"]]

        # 计算各个k值的指标
        metrics = {}

        for k in self.k_values:
            # Recall@k
            recall_k = self.calculate_recall_at_k(retrieved_indices, correct_answer_indices, k)
            metrics[f"recall@{k}"] = recall_k

            # NDCG@k
            ndcg_k = self.calculate_ndcg_at_k(retrieved_indices, correct_answer_indices, k, relevance_scores)
            metrics[f"ndcg@{k}"] = ndcg_k

        return {
            "query": query,
            "correct_answers": correct_answers,
            "correct_answer_indices": correct_answer_indices,
            "found_answers_count": len(correct_answer_indices),
            "total_answers_count": len(correct_answers),
            "retrieved_indices": retrieved_indices[:self.max_k],
            "metrics": metrics,
            "search_results": search_results["results"][:self.max_k]
        }

    def evaluate_single_query_worker(self, query: str, correct_answers: List[str],
                                     relevance_scores: Optional[List[float]] = None) -> Dict:
        """
        单个查询评估的工作函数（线程安全）

        Args:
            query: 查询文本
            correct_answers: 正确答案列表
            relevance_scores: 相关性分数（可选）

        Returns:
            评估结果
        """
        # 使用锁保护向量库访问
        with self.model_lock:
            result = self.evaluate_single_query(query, correct_answers, relevance_scores)
        return result

    def evaluate_batch(self, test_cases: Dict[str, List[str]],
                       relevance_scores: Optional[Dict[str, List[float]]] = None,
                       show_progress: bool = True,
                       use_parallel: bool = True) -> Dict:
        """
        批量评估查询（支持并行）

        Args:
            test_cases: 测试案例字典 {question: [answer1, answer2, ...]}
            relevance_scores: 相关性分数字典（可选）
            show_progress: 是否显示进度条
            use_parallel: 是否使用并行评估（默认True）

        Returns:
            评估结果字典
        """
        queries = list(test_cases.keys())
        print(f"🔄 开始批量评估 {len(queries)} 个查询...")
        print(f"⚙️  并行模式: {'开启' if use_parallel else '关闭'}")

        # 存储所有结果
        all_results = []
        metric_sums = defaultdict(float)

        # 统计信息
        total_answers = 0
        found_answers = 0
        zero_recall_queries = 0

        if use_parallel and self.max_workers > 1:
            # 并行评估
            print(f"🧵 使用 {self.max_workers} 个线程并行评估")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_query = {}
                for query in queries:
                    correct_answers = test_cases[query]
                    scores = relevance_scores.get(query) if relevance_scores else None
                    future = executor.submit(
                        self.evaluate_single_query_worker,
                        query,
                        correct_answers,
                        scores
                    )
                    future_to_query[future] = query

                # 收集结果（使用进度条）
                if show_progress:
                    iterator = tqdm(as_completed(future_to_query), total=len(queries), desc="评估查询")
                else:
                    iterator = as_completed(future_to_query)

                for future in iterator:
                    try:
                        result = future.result()
                        all_results.append(result)

                        # 累计指标
                        for metric_name, value in result["metrics"].items():
                            metric_sums[metric_name] += value

                        # 统计信息
                        total_answers += result["total_answers_count"]
                        found_answers += result["found_answers_count"]
                        if result["metrics"]["recall@1"] == 0.0:
                            zero_recall_queries += 1

                    except Exception as e:
                        print(f"\n❌ 查询评估失败: {e}")

        else:
            # 串行评估（原有逻辑）
            iterator = tqdm(queries, desc="评估查询") if show_progress else queries

            for query in iterator:
                correct_answers = test_cases[query]
                scores = relevance_scores.get(query) if relevance_scores else None

                # 评估单个查询
                result = self.evaluate_single_query(query, correct_answers, scores)
                all_results.append(result)

                # 累计指标
                for metric_name, value in result["metrics"].items():
                    metric_sums[metric_name] += value

                # 统计信息
                total_answers += result["total_answers_count"]
                found_answers += result["found_answers_count"]
                if result["metrics"]["recall@1"] == 0.0:
                    zero_recall_queries += 1

        # 计算平均指标
        num_queries = len(queries)
        avg_metrics = {metric: total / num_queries for metric, total in metric_sums.items()}

        # 计算答案覆盖率
        answer_coverage = found_answers / total_answers if total_answers > 0 else 0.0

        # 准备最终结果
        evaluation_result = {
            "summary": {
                "total_queries": num_queries,
                "total_answers": total_answers,
                "found_answers": found_answers,
                "answer_coverage": answer_coverage,
                "zero_recall_queries": zero_recall_queries,
                "zero_recall_percentage": zero_recall_queries / num_queries * 100,
                "avg_metrics": avg_metrics,
                "k_values": self.k_values
            },
            "detailed_results": all_results
        }

        # 打印结果摘要
        self._print_evaluation_summary(evaluation_result["summary"])

        return evaluation_result

    def _print_evaluation_summary(self, summary: Dict):
        """打印评估结果摘要"""
        print("\n" + "=" * 60)
        print("📊 评估结果摘要（基于字符串匹配）")
        print("=" * 60)

        # 基本统计
        print("📈 基本统计:")
        print(f"   总查询数: {summary['total_queries']}")
        print(f"   总答案数: {summary['total_answers']}")
        print(f"   找到答案数: {summary['found_answers']}")
        print(f"   答案覆盖率: {summary['answer_coverage']:.4f} ({summary['answer_coverage'] * 100:.2f}%)")
        print(f"   零召回查询数: {summary['zero_recall_queries']} ({summary['zero_recall_percentage']:.1f}%)")

        # 分别打印Recall和NDCG
        print("\n🎯 Recall指标:")
        for k in self.k_values:
            recall = summary['avg_metrics'][f"recall@{k}"]
            print(f"   Recall@{k}: {recall:.4f} ({recall * 100:.2f}%)")

        print("\n📈 NDCG指标:")
        for k in self.k_values:
            ndcg = summary['avg_metrics'][f"ndcg@{k}"]
            print(f"   NDCG@{k}: {ndcg:.4f}")

        print("=" * 60)

    def load_test_cases(self, test_cases_path: str) -> Dict[str, List[str]]:
        """
        从文件加载测试案例

        Args:
            test_cases_path: 测试案例文件路径

        Returns:
            测试案例字典
        """
        print(f"📂 加载测试案例: {test_cases_path}")

        try:
            with open(test_cases_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)

            print(f"✅ 成功加载 {len(test_cases)} 个测试案例")

            # 验证格式
            for question, answers in test_cases.items():
                if not isinstance(answers, list):
                    raise ValueError(f"问题 '{question}' 的答案不是列表格式")
                if not answers:
                    print(f"⚠️  问题 '{question}' 没有答案")

            return test_cases

        except Exception as e:
            raise Exception(f"加载测试案例失败: {e}")

    def save_results(self, evaluation_result: Dict, save_path: str,
                     save_format: str = "json"):
        """
        保存评估结果

        Args:
            evaluation_result: 评估结果字典
            save_path: 保存路径（不包含扩展名）
            save_format: 保存格式 ("json" 或 "csv")
        """
        if save_format == "json":
            # 保存为JSON格式
            with open(f"{save_path}.json", 'w', encoding='utf-8') as f:
                json_result = self._convert_for_json(evaluation_result)
                json.dump(json_result, f, ensure_ascii=False, indent=2)

        elif save_format == "csv":
            # 保存摘要为CSV格式
            summary_df = pd.DataFrame([evaluation_result["summary"]["avg_metrics"]])
            summary_df.to_csv(f"{save_path}_summary.csv", index=False)

            # 保存详细结果为CSV格式
            detailed_data = []
            for result in evaluation_result["detailed_results"]:
                row = {
                    "query": result["query"],
                    "correct_answers_count": result["total_answers_count"],
                    "found_answers_count": result["found_answers_count"],
                    "answer_coverage": result["found_answers_count"] / result["total_answers_count"] if result[
                                                                                                             "total_answers_count"] > 0 else 0
                }
                row.update(result["metrics"])
                detailed_data.append(row)

            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(f"{save_path}_detailed.csv", index=False)

        print(f"✅ 评估结果已保存到: {save_path}.{save_format}")

    def _convert_for_json(self, obj):
        """转换对象为JSON可序列化格式"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                result[key] = self._convert_for_json(value)
            return result
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def analyze_failure_cases(self, evaluation_result: Dict,
                               threshold: float = 0.3) -> List[Dict]:
        """
        分析失败案例（Recall@1较低的查询）

        Args:
            evaluation_result: 评估结果
            threshold: Recall@1阈值，低于此值的被认为是失败案例

        Returns:
            失败案例列表
        """
        failure_cases = []

        for result in evaluation_result["detailed_results"]:
            recall_1 = result["metrics"]["recall@1"]
            if recall_1 < threshold:
                # 获取检索到的前3个文本
                top_retrieved_texts = []
                for search_result in result["search_results"][:3]:
                    top_retrieved_texts.append({
                        "text": search_result["text"][:100] + "...",  # 截断显示
                        "score": search_result["score"]
                    })

                failure_case = {
                    "query": result["query"],
                    "recall@1": recall_1,
                    "found_answers_count": result["found_answers_count"],
                    "total_answers_count": result["total_answers_count"],
                    "top_3_retrieved": top_retrieved_texts
                }
                failure_cases.append(failure_case)

        # 按recall@1升序排序
        failure_cases.sort(key=lambda x: x["recall@1"])

        print(f"🔍 发现 {len(failure_cases)} 个失败案例 (Recall@1 < {threshold})")

        return failure_cases

    def get_missing_answers_report(self, test_cases: Dict[str, List[str]]) -> Dict:
        """
        生成缺失答案报告（基于字符串匹配）

        Args:
            test_cases: 测试案例字典

        Returns:
            缺失答案报告
        """
        print("🔍 分析缺失答案（基于字符串匹配）...")

        missing_answers = []
        answer_frequency = defaultdict(int)

        for answers in test_cases.values():
            for answer in answers:
                answer_frequency[answer] += 1
                if answer not in self.vector_db.texts:
                    missing_answers.append(answer)

        # 统计唯一缺失答案
        unique_missing = set(missing_answers)

        # 按频率排序缺失的答案
        missing_with_freq = [(ans, answer_frequency[ans]) for ans in unique_missing]
        missing_with_freq.sort(key=lambda x: x[1], reverse=True)

        report = {
            "total_unique_answers": len(answer_frequency),
            "missing_answers_count": len(unique_missing),
            "missing_percentage": len(unique_missing) / len(answer_frequency) * 100 if len(
                answer_frequency) > 0 else 0,
            "most_frequent_missing": missing_with_freq[:5]  # 最常出现的缺失答案
        }

        print(f"   总答案数: {report['total_unique_answers']}")
        print(f"   缺失答案数: {report['missing_answers_count']}")
        print(f"   缺失比例: {report['missing_percentage']:.2f}%")

        if report['most_frequent_missing']:
            print("   最常见的缺失答案:")
            for ans, freq in report['most_frequent_missing']:
                print(f"     '{ans[:50]}...' (出现{freq}次)")

        return report
