#!/usr/bin/env python3
"""
批量处理器 - 使用多线程并行处理多个工具的案例生成

采用多线程方案：
- 更适合 I/O 密集型任务（LLM API 调用）
- 可以开更多并发（50-100 个线程）
- 内存占用低
- 完美支持 tqdm 进度条
- 使用线程安全的 stdout 重定向，确保每个线程的日志正确分离
"""

import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from ..base import BaseProcessor
from ..models import Tool, Toolbox, ProcessResult, ProcessStatus
from ..workflows.stage1_orchestrator import Stage1Orchestrator
from ..config import LLMConfig


class ThreadLocalStdout:
    """
    线程安全的 stdout 包装器

    每个线程可以设置自己的输出文件，互不干扰
    """

    def __init__(self, original_stdout):
        self._original_stdout = original_stdout
        self._local = threading.local()
        self._lock = threading.Lock()

    def set_file(self, file_obj):
        """为当前线程设置输出文件"""
        self._local.file = file_obj

    def clear_file(self):
        """清除当前线程的输出文件设置"""
        if hasattr(self._local, 'file'):
            delattr(self._local, 'file')

    def write(self, text):
        """写入文本到当前线程的输出目标"""
        # 获取当前线程的文件对象
        file_obj = getattr(self._local, 'file', None)

        if file_obj is not None:
            # 写入线程专属的文件
            file_obj.write(text)
        else:
            # 写入原始 stdout
            self._original_stdout.write(text)

    def flush(self):
        """刷新输出缓冲"""
        file_obj = getattr(self._local, 'file', None)

        if file_obj is not None:
            file_obj.flush()
        else:
            self._original_stdout.flush()

    def fileno(self):
        """返回文件描述符（某些库需要）"""
        return self._original_stdout.fileno()

    def isatty(self):
        """检查是否为终端"""
        file_obj = getattr(self._local, 'file', None)
        if file_obj is not None and hasattr(file_obj, 'isatty'):
            return file_obj.isatty()
        return self._original_stdout.isatty()


class BatchProcessor(BaseProcessor):
    """
    批量处理器 - 使用多线程并行处理多个工具

    继承 BaseProcessor，实现具体的批量处理逻辑
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        output_dir: str,
        max_workers: Optional[int] = None,
        retry_attempts: int = 2,
        num_scenarios: int = 2,
        cases_per_scenario: int = 2,
        with_parameters: bool = False,
        enable_revision: bool = True,
        verbose: bool = False
    ):
        """
        初始化批量处理器

        Args:
            llm_config: LLM配置
            output_dir: 输出目录
            max_workers: 最大并发线程数（None=自动，推荐50）
            retry_attempts: 重试次数
            num_scenarios: 每个工具生成的场景数（默认2，用于快速测试）
            cases_per_scenario: 每个场景生成的案例数（默认2，用于快速测试）
            with_parameters: 是否在案例生成时包含工具参数
            enable_revision: 是否启用案例修正
            verbose: 是否显示详细日志
        """
        # 初始化基类
        super().__init__(max_workers=max_workers, retry_attempts=retry_attempts)

        # 如果没有指定，使用50个线程（多线程可以开更多）
        if self.max_workers is None:
            self.max_workers = 50

        self.llm_config = llm_config
        self.output_dir = output_dir
        self.num_scenarios = num_scenarios
        self.cases_per_scenario = cases_per_scenario
        self.with_parameters = with_parameters
        self.enable_revision = enable_revision
        self.verbose = verbose

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建日志目录
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 线程锁（保护日志文件写入）
        self.log_lock = threading.Lock()

        # 安装线程安全的 stdout/stderr 包装器
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._thread_stdout = ThreadLocalStdout(self._original_stdout)
        self._thread_stderr = ThreadLocalStdout(self._original_stderr)

        # 替换全局的 stdout 和 stderr
        sys.stdout = self._thread_stdout
        sys.stderr = self._thread_stderr

        print(f"🚀 批量处理器初始化完成")
        print(f"📊 并发配置: 多线程模式，{self.max_workers} 个线程（线程安全日志）")
        print(f"📁 输出目录: {output_dir}")
        print(f"📝 日志目录: {self.log_dir}")
        print(f"🔄 重试次数: {retry_attempts}")
        print(f"📋 案例配置: {num_scenarios}场景 × {cases_per_scenario}案例 = {num_scenarios * cases_per_scenario}总案例")

    def __del__(self):
        """析构函数：恢复原始的 stdout/stderr"""
        try:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        except:
            pass

    def _create_llm_instances(self):
        """
        创建 LLM 实例（每个线程独立创建）

        虽然 LLM 是线程安全的，但为了稳妥，每个任务创建独立实例

        Returns:
            包含四个 LLM 实例的字典
        """
        from langchain_openai import ChatOpenAI

        return {
            'scenario_llm': ChatOpenAI(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=0.6,  # 场景生成
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout
            ),
            'case_llm': ChatOpenAI(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=0.6,  # 案例生成
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout
            ),
            'check_llm': ChatOpenAI(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=0.3,  # 案例验证（低温度，一致性判断）
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout
            ),
            'revise_llm': ChatOpenAI(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=0.7,  # 案例修正（高温度，创造性）
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout
            )
        }

    def process_single_task(self, task_data: Dict[str, Any]) -> ProcessResult:
        """
        处理单个工具（子类必须实现）

        Args:
            task_data: 任务数据，包含 tool 和 toolbox

        Returns:
            ProcessResult对象
        """
        tool_dict = task_data['tool']
        toolbox_dict = task_data['toolbox']

        # 转换为数据模型
        tool = Tool.from_dict(tool_dict)
        toolbox = Toolbox.from_dict(toolbox_dict)

        # 创建日志文件
        tool_id = tool.algorithm_id.replace(':', '_')
        log_file = os.path.join(self.log_dir, f"{tool_id}.log")

        start_time = time.time()

        try:
            # 每个线程创建独立的 LLM 实例
            llm_instances = self._create_llm_instances()

            # 创建工作流编排器
            orchestrator = Stage1Orchestrator(
                scenario_llm=llm_instances['scenario_llm'],
                case_llm=llm_instances['case_llm'],
                check_llm=llm_instances['check_llm'],
                revise_llm=llm_instances['revise_llm'],
                output_dir=self.output_dir,
                num_scenarios=self.num_scenarios,
                cases_per_scenario=self.cases_per_scenario,
                with_parameters=self.with_parameters,
                enable_revision=self.enable_revision
            )

            # 如果不是 verbose 模式，使用线程安全的日志重定向
            if not self.verbose:
                with open(log_file, 'w', encoding='utf-8') as log_f:
                    # 为当前线程设置专属的输出文件
                    self._thread_stdout.set_file(log_f)
                    self._thread_stderr.set_file(log_f)

                    try:
                        # 运行工作流（输出会自动写入当前线程的日志文件）
                        workflow_result = orchestrator.run(tool=tool, toolbox=toolbox)
                    finally:
                        # 清除当前线程的文件设置
                        self._thread_stdout.clear_file()
                        self._thread_stderr.clear_file()
            else:
                # verbose 模式，直接输出到原始 stdout（需要加锁避免混乱）
                with self.log_lock:
                    workflow_result = orchestrator.run(tool=tool, toolbox=toolbox)

            elapsed_time = time.time() - start_time

            # 构建 ProcessResult（使用正确的参数名）
            return ProcessResult(
                status=workflow_result.status,
                tool_id=tool.algorithm_id,
                tool_name=tool.name,
                toolbox_name=toolbox.name,
                duration_minutes=elapsed_time / 60.0,
                log_file=log_file,
                result=workflow_result.to_dict(),
                error=workflow_result.error_message,
                is_api_error=False
            )

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)

            # 记录错误到日志
            with self.log_lock:
                try:
                    with open(log_file, 'a', encoding='utf-8') as log_f:
                        log_f.write(f"\n{'='*80}\n")
                        log_f.write(f"❌ 工具处理失败: {tool.name}\n")
                        log_f.write(f"错误信息: {error_msg}\n")
                        log_f.write(f"耗时: {elapsed_time:.2f}秒\n")
                        log_f.write(f"{'='*80}\n")
                except:
                    pass

            return ProcessResult(
                status=ProcessStatus.FAILED,
                tool_id=tool.algorithm_id,
                tool_name=tool.name,
                toolbox_name=toolbox.name,
                duration_minutes=elapsed_time / 60.0,
                log_file=log_file,
                result=None,
                error=error_msg,
                is_api_error=self.is_api_error(error_msg)
            )

    def run(
        self,
        tools_file: str,
        start_from: int = 0,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行批量处理（子类必须实现）

        Args:
            tools_file: 工具文件路径
            start_from: 从第几个工具开始（用于断点续传）
            limit: 限制处理工具数量（用于测试）

        Returns:
            统计信息
        """
        print(f"\n{'='*80}")
        print(f"🚀 开始批量处理")
        print(f"{'='*80}")
        print(f"📄 工具文件: {tools_file}")

        # 加载工具数据
        tools_data = self._load_tools_file(tools_file)

        # 切片处理
        if limit:
            tools_data = tools_data[start_from:start_from + limit]
            print(f"📋 处理工具: {start_from} ~ {start_from + len(tools_data)} (共 {len(tools_data)} 个)")
        else:
            tools_data = tools_data[start_from:]
            print(f"📋 处理工具: {start_from} ~ 结束 (共 {len(tools_data)} 个)")

        total_tools = len(tools_data)
        if total_tools == 0:
            print("⚠️  没有需要处理的工具")
            return {'total_tools': 0, 'successful': 0, 'failed': 0}

        print(f"🔧 并发线程数: {self.max_workers}")
        print(f"{'='*80}\n")

        # 开始计时
        self.start_timer()

        # 使用 ThreadPoolExecutor 并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.process_single_task, tool_data): tool_data
                for tool_data in tools_data
            }

            # 使用 tqdm 显示进度条（自动处理多线程）
            with tqdm(
                total=total_tools,
                desc="处理工具",
                ncols=100,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ) as pbar:
                for future in as_completed(futures):
                    tool_data = futures[future]
                    tool_dict = tool_data['tool']

                    try:
                        result = future.result()

                        # 跟踪结果
                        self.track_result(result)

                        # 更新进度条
                        if result.status == ProcessStatus.SUCCESS:
                            pbar.set_postfix_str(f"✅ {tool_dict['name']}")
                        else:
                            pbar.set_postfix_str(f"❌ {tool_dict['name']}")

                        pbar.update(1)

                    except Exception as e:
                        # 进程级别的异常
                        pbar.set_postfix_str(f"💥 {tool_dict['name']}: {e}")
                        pbar.update(1)

        # 停止计时
        self.stop_timer()

        # 获取统计信息
        stats = self.get_statistics(
            total_count=total_tools,
            execution_time=self.get_elapsed_time()
        )

        # 保存失败结果
        self.save_results(self.output_dir, prefix="batch_")

        # 打印统计
        self.print_statistics(stats)

        # 打印永久失败的工具提示
        if self.permanently_failed_results:
            print(f"\n⚠️  提示: {len(self.permanently_failed_results)} 个工具重试{self.retry_attempts}次后仍失败")
            print(f"   详情请查看: {self.output_dir}/batch_permanently_failed_*.json")

        return stats

    def _load_tools_file(self, tools_file: str) -> List[Dict]:
        """
        加载工具文件并解析

        支持两种格式：
        1. {"selected_tools": [{tool, toolbox}, ...]}
        2. [{toolbox1}, {toolbox2}, ...] (需要展开)

        Args:
            tools_file: 工具文件路径

        Returns:
            工具数据列表 [{tool, toolbox}, ...]
        """
        with open(tools_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 格式1：包装格式
        if isinstance(data, dict) and 'selected_tools' in data:
            return data['selected_tools']

        # 格式2：工具箱数组格式，需要展开
        elif isinstance(data, list):
            tools_data = []
            for toolbox in data:
                toolbox_info = {
                    "name": toolbox.get("toolbox", toolbox.get("name", "")),
                    "description": toolbox.get("discription", toolbox.get("description", ""))
                }
                for tool in toolbox.get("tools", []):
                    tools_data.append({
                        "tool": tool,
                        "toolbox": toolbox_info
                    })
            return tools_data

        else:
            raise ValueError("不支持的工具文件格式")
