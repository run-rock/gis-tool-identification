#!/usr/bin/env python3
"""
第二阶段批处理器

使用多线程并行处理多个工具的相似工具搜索和评估
"""

import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from langchain_openai import ChatOpenAI

from ..base import BaseProcessor
from ..config import LLMConfig, EmbeddingConfig, ModelType
from ..services.embedding_service import EmbeddingService
from ..services.vector_db import VectorDatabase
from ..workflows.stage2_orchestrator import Stage2Orchestrator
from ..models import ProcessResult, ProcessStatus


class Stage2BatchProcessor(BaseProcessor):
    """
    第二阶段批处理器

    负责批量处理第一阶段生成的案例文件：
    1. 为每个案例搜索相似工具（使用向量数据库）
    2. 两阶段评估相似工具
    3. 生成最终结果
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        vector_db_path: str,
        input_dir: str,
        output_dir: str,
        max_workers: Optional[int] = None,
        retry_attempts: int = 2,
        top_k: int = 3,
        verbose: bool = False
    ):
        """
        初始化第二阶段批处理器

        Args:
            llm_config: LLM配置
            embedding_config: Embedding配置
            vector_db_path: 向量数据库路径（不含扩展名）
            input_dir: 输入目录（包含final_cases_*.json文件）
            output_dir: 输出目录
            max_workers: 最大线程数
            retry_attempts: 重试次数
            top_k: 每个案例搜索的相似工具数量
            verbose: 是否显示详细日志
        """
        super().__init__(max_workers=max_workers, retry_attempts=retry_attempts)

        self.llm_config = llm_config
        self.embedding_config = embedding_config
        self.vector_db_path = vector_db_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.top_k = top_k
        self.verbose = verbose

        # 使用线程池，默认并发数可以设置更高
        if self.max_workers is None:
            import multiprocessing as mp
            self.max_workers = min(50, max(10, mp.cpu_count() * 2))

        # 创建输出目录（与旧代码一致的目录结构）
        os.makedirs(output_dir, exist_ok=True)

        # 创建子目录（与旧代码保持一致）
        self.tool_evaluations_dir = os.path.join(output_dir, "tool_evaluations")
        os.makedirs(self.tool_evaluations_dir, exist_ok=True)

        # 共享资源（预加载）
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_db: Optional[VectorDatabase] = None

        # 线程锁（用于日志写入）
        self.log_lock = threading.Lock()

        print(f"🚀 第二阶段批处理器初始化")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🧵 最大线程数: {self.max_workers}")
        print(f"🔍 每案例搜索工具数: {top_k}")

    def _load_shared_resources(self):
        """预加载共享的Embedding模型和向量数据库"""
        print("\n" + "="*80)
        print("🔄 加载共享资源...")
        print("="*80)

        # 1. 加载Embedding服务
        print("📦 加载Embedding模型...")
        self.embedding_service = EmbeddingService(self.embedding_config)

        # 2. 加载向量数据库
        print("📚 加载向量数据库...")
        self.vector_db = VectorDatabase(
            embedding_service=self.embedding_service,
            normalize_embeddings=True
        )
        self.vector_db.load_from_disk(self.vector_db_path)

        print("✅ 所有共享资源加载完成\n")
        print("="*80)

    def _create_llm_instance(self):
        """创建LLM实例（每个线程独立创建）"""
        import os

        # 如果 api_key 为 None，尝试从环境变量读取
        api_key = self.llm_config.api_key
        if api_key is None:
            api_key = os.getenv("QWEN3_KEY") or os.getenv("OPENAI_API_KEY")

        return ChatOpenAI(
            model=self.llm_config.model,
            api_key=api_key,
            base_url=self.llm_config.base_url,
            temperature=0.3,  # 第二阶段评估用低温度
            max_tokens=self.llm_config.max_tokens,
            timeout=self.llm_config.timeout
        )

    def _find_case_files(self) -> List[str]:
        """查找所有final_cases_*.json文件"""
        case_files = []

        if not os.path.exists(self.input_dir):
            print(f"❌ 输入目录不存在: {self.input_dir}")
            return case_files

        for file in os.listdir(self.input_dir):
            if file.startswith("final_cases_") and file.endswith(".json"):
                file_path = os.path.join(self.input_dir, file)
                case_files.append(file_path)

        return case_files

    def process_single_task(self, task_data: Dict[str, Any]) -> ProcessResult:
        """
        处理单个工具（覆盖BaseProcessor的抽象方法）

        Args:
            task_data: 任务数据，包含：
                - case_file: 案例文件路径
                - output_file: 输出文件路径

        Returns:
            ProcessResult
        """
        case_file = task_data['case_file']
        output_file = task_data['output_file']
        tool_name = os.path.basename(case_file).replace("final_cases_", "").replace(".json", "")

        start_time = datetime.now()

        try:
            # 创建独立的LLM实例
            llm = self._create_llm_instance()

            # 创建第二阶段编排器（使用共享的向量数据库）
            orchestrator = Stage2Orchestrator(
                vector_db=self.vector_db,
                llm=llm,
                top_k=self.top_k,
                max_retries=self.retry_attempts
            )

            # 运行第二阶段处理
            report = orchestrator.run(case_file, output_file)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return ProcessResult(
                status=ProcessStatus.SUCCESS,
                task_id=tool_name,
                message=f"处理完成，添加 {report['summary']['total_similar_tools_added']} 个相似工具",
                duration_seconds=duration,
                metadata={
                    'input_file': case_file,
                    'output_file': output_file,
                    'total_cases': report['summary']['total_cases'],
                    'total_similar_tools_added': report['summary']['total_similar_tools_added']
                }
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = str(e)
            is_api_error = self._is_api_error(error_msg)

            return ProcessResult(
                status=ProcessStatus.FAILED,
                task_id=tool_name,
                message=f"处理失败: {error_msg}",
                duration_seconds=duration,
                error=error_msg,
                metadata={
                    'input_file': case_file,
                    'is_api_error': is_api_error
                }
            )

    def _is_api_error(self, error_msg: str) -> bool:
        """判断是否为API错误"""
        api_error_keywords = [
            'rate limit', 'Rate limit', 'RATE_LIMIT',
            'api key', 'API key', 'API_KEY',
            'quota', 'Quota', 'QUOTA',
            'timeout', 'Timeout', 'TIMEOUT',
            'connection', 'Connection', 'CONNECTION',
            'network', 'Network', 'NETWORK',
            '429', '503', '502', '500',
            'Service Unavailable',
            'Bad Gateway',
            'Internal Server Error'
        ]
        return any(keyword in error_msg for keyword in api_error_keywords)

    def run(
        self,
        start_from: int = 0,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行批处理

        Args:
            start_from: 从第几个工具开始处理
            limit: 限制处理数量

        Returns:
            统计结果
        """
        print("\n🚀 开始第二阶段批量处理")

        # 预加载共享资源
        self._load_shared_resources()

        # 查找所有案例文件
        case_files = self._find_case_files()

        if not case_files:
            print("❌ 没有找到需要处理的案例文件")
            return {
                'total_tools': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'execution_time': 0
            }

        # 应用 start_from 和 limit
        case_files = case_files[start_from:]
        if limit is not None:
            case_files = case_files[:limit]

        total_tools = len(case_files)
        print(f"📋 总工具数: {total_tools}")
        print(f"🧵 并行线程数: {self.max_workers}")

        # 准备任务数据
        tasks = []
        for case_file in case_files:
            file_name = os.path.basename(case_file)
            # 输出文件放在 tool_evaluations/ 子目录，不带 _evaluated 后缀（与旧代码一致）
            output_file = os.path.join(
                self.tool_evaluations_dir,
                file_name  # 保持原文件名，不加 _evaluated
            )
            tasks.append({
                'case_file': case_file,
                'output_file': output_file
            })

        # 使用ThreadPoolExecutor并行处理
        start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.process_single_task, task): task
                for task in tasks
            }

            print(f"📤 已提交 {len(futures)} 个任务到线程池")
            print(f"🚀 开始处理...")

            # 处理完成的任务（带进度条）
            with tqdm(total=total_tools, desc="处理工具", ncols=100) as pbar:
                for future in as_completed(futures):
                    task = futures[future]
                    tool_name = os.path.basename(task['case_file']).replace("final_cases_", "").replace(".json", "")

                    try:
                        result = future.result()
                        self.track_result(result)

                        if result.status == ProcessStatus.SUCCESS:
                            pbar.set_postfix_str(f"✅ {tool_name}")
                        else:
                            pbar.set_postfix_str(f"❌ {tool_name}")

                    except Exception as e:
                        # 线程执行异常
                        error_result = ProcessResult(
                            status=ProcessStatus.FAILED,
                            task_id=tool_name,
                            message=f"线程执行异常: {str(e)}",
                            error=str(e),
                            metadata={'case_file': task['case_file']}
                        )
                        self.track_result(error_result)
                        pbar.set_postfix_str(f"💥 {tool_name}")

                    pbar.update(1)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # 打印统计结果
        stats = self.get_statistics()
        stats['execution_time'] = execution_time
        stats['avg_time_per_tool'] = execution_time / total_tools if total_tools > 0 else 0

        self.print_statistics()
        print(f"⏱️  总执行时间: {execution_time:.1f}秒")
        print(f"📊 平均每工具: {stats['avg_time_per_tool']:.1f}秒")

        # 保存处理结果（与旧代码格式一致）
        self._save_all_results()

        return stats

    def _save_all_results(self):
        """保存所有处理结果（与旧代码格式一致）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存成功结果
        if self.successful_results:
            success_file = os.path.join(self.output_dir, f"successful_tools_{timestamp}.json")
            with open(success_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'total_successful': len(self.successful_results),
                    'successful_tools': [r.to_dict() for r in self.successful_results]
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 成功工具记录保存至: {success_file}")

        # 保存所有失败结果
        if self.failed_results:
            failed_file = os.path.join(self.output_dir, f"all_failed_tools_{timestamp}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'total_failed': len(self.failed_results),
                    'failed_tools': [r.to_dict() for r in self.failed_results]
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 所有失败工具保存至: {failed_file}")

        # 保存永久失败结果（重试多次后仍失败）
        if self.permanently_failed_results:
            perm_failed_file = os.path.join(self.output_dir, f"permanently_failed_tools_{timestamp}.json")
            with open(perm_failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'max_retries': self.retry_attempts,
                    'total_permanently_failed': len(self.permanently_failed_results),
                    'permanently_failed_tools': [r.to_dict() for r in self.permanently_failed_results],
                    'note': f'这些工具已重试{self.retry_attempts}次后仍失败'
                }, f, ensure_ascii=False, indent=2)
            print(f"⛔ 永久失败工具保存至: {perm_failed_file}")
            print(f"   ({len(self.permanently_failed_results)} 个工具重试{self.retry_attempts}次后仍失败)")
