#!/usr/bin/env python3
"""
基类定义 - 抽象公共逻辑
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .models import ProcessResult, ProcessStatus


# ============================================================================
# Mixin 类定义
# ============================================================================

class LoggerMixin:
    """
    日志混入类 - 提供统一的日志功能

    可以被任何类混入使用
    """

    @staticmethod
    def create_log_file(log_dir: str, name: str) -> str:
        """
        创建日志文件

        Args:
            log_dir: 日志目录
            name: 日志名称

        Returns:
            日志文件路径
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        return log_file

    @staticmethod
    def write_log(log_file: str, message: str, mode: str = 'a'):
        """
        写入日志

        Args:
            log_file: 日志文件路径
            message: 日志消息
            mode: 写入模式（'a' 追加，'w' 覆盖）
        """
        with open(log_file, mode, encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    @staticmethod
    def write_log_header(log_file: str, title: str, info: Dict[str, str]):
        """
        写入日志头部

        Args:
            log_file: 日志文件路径
            title: 标题
            info: 信息字典
        """
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"{title}\n")
            f.write("="*80 + "\n")
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
            f.write("="*80 + "\n\n")


class TimerMixin:
    """
    计时混入类 - 提供统一的计时功能
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start_timer(self):
        """开始计时"""
        self._start_time = time.time()

    def stop_timer(self):
        """停止计时"""
        self._end_time = time.time()

    def get_elapsed_time(self) -> float:
        """
        获取经过的时间（秒）

        Returns:
            经过的秒数
        """
        if self._start_time is None:
            return 0.0

        end = self._end_time if self._end_time else time.time()
        return end - self._start_time

    def get_elapsed_minutes(self) -> float:
        """
        获取经过的时间（分钟）

        Returns:
            经过的分钟数
        """
        return self.get_elapsed_time() / 60


# ============================================================================
# 抽象基类定义
# ============================================================================

# 在 Python 中，如果一个类继承了 ABC (Abstract Base Class) 并且包含 @abstractmethod，它就强制不能被实例化。
"""
from abc import ABC, abstractmethod

# 这是一个基类（抽象类）
class BaseProcessor(ABC): 
    @abstractmethod
    def process(self):
        pass

# ❌ 错误做法：直接实例化基类
# processor = BaseProcessor() 
# 报错：TypeError: Can't instantiate abstract class...
# 因为 process 方法里是空的，没有任何代码，创建出来也没法跑。

# ✅ 正确做法：实例化它的子类（成品）
class ImageProcessor(BaseProcessor):
    def process(self):
        print("正在处理图片...")

p = ImageProcessor() # 这是可以的
p.process()
"""

# 类的init方法中的类变量，都相当于状态
class BaseProcessor(ABC, LoggerMixin, TimerMixin):
    """
    处理器基类 - 抽象并行处理的公共逻辑

    所有并行处理器（Stage1, Stage2等）都继承此类
    混入LoggerMixin和TimerMixin提供日志记录和计时功能
    """

    def __init__(self, max_workers: Optional[int] = None, retry_attempts: int = 2):
        """
        初始化处理器

        Args:
            max_workers: 最大并行数
            retry_attempts: 重试次数
        """
        TimerMixin.__init__(self)  # 初始化TimerMixin
        self.max_workers = max_workers or self._default_workers()
        self.retry_attempts = retry_attempts

        # 结果跟踪（统一管理，避免重复定义）
        self.successful_results: List[ProcessResult] = []
        self.failed_results: List[ProcessResult] = []
        self.api_failed_results: List[ProcessResult] = []
        self.permanently_failed_results: List[ProcessResult] = []
        self.retry_counts: Dict[str, int] = {}

        print(f"🚀 {self.__class__.__name__} 初始化")
        print(f"📊 最大并行数: {self.max_workers}")
        print(f"🔄 重试次数: {retry_attempts}")

    @staticmethod
    def _default_workers() -> int:
        """计算默认并行数"""
        import multiprocessing as mp
        return max(1, int(mp.cpu_count() * 0.75))

    @staticmethod
    def is_api_error(error_msg: str) -> bool:
        """
        判断是否为API相关错误

        Args:
            error_msg: 错误信息

        Returns:
            是否为API错误
        """
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

    def track_result(self, result: ProcessResult):
        """
        跟踪处理结果（统一的结果管理）

        Args:
            result: 处理结果
        """
        if result.status == ProcessStatus.SUCCESS:
            self.successful_results.append(result)
            # 成功后清除重试计数
            if result.tool_id in self.retry_counts:
                del self.retry_counts[result.tool_id]
        else:
            # 记录重试次数
            self.retry_counts[result.tool_id] = self.retry_counts.get(result.tool_id, 0) + 1
            result.retry_count = self.retry_counts[result.tool_id]

            # 如果超过重试次数，标记为永久失败
            if result.retry_count > self.retry_attempts:
                self.permanently_failed_results.append(result)

            self.failed_results.append(result)

            # API错误单独跟踪
            if result.is_api_error:
                self.api_failed_results.append(result)

    def get_statistics(self, total_count: int, execution_time: float) -> Dict[str, Any]:
        """
        获取统计信息（统一的统计逻辑）

        Args:
            total_count: 总任务数
            execution_time: 执行时间（秒）

        Returns:
            统计字典
        """
        return {
            'total': total_count,
            'successful': len(self.successful_results),
            'failed': len(self.failed_results),
            'api_failed': len(self.api_failed_results),
            'permanently_failed': len(self.permanently_failed_results),
            'success_rate': len(self.successful_results) / total_count * 100 if total_count > 0 else 0,
            'execution_time_seconds': execution_time,
            'execution_time_minutes': execution_time / 60,
            'avg_time_per_task': execution_time / total_count if total_count > 0 else 0
        }

    def save_results(self, output_dir: str, prefix: str = ""):
        """
        保存处理结果（统一的保存逻辑）

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存成功结果
        if self.successful_results:
            success_file = os.path.join(output_dir, f"{prefix}successful_{timestamp}.json")
            with open(success_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'total': len(self.successful_results),
                    'results': [r.to_dict() for r in self.successful_results]
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 成功结果保存至: {success_file}")

        # 保存失败结果
        if self.failed_results:
            failed_file = os.path.join(output_dir, f"{prefix}failed_{timestamp}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'total': len(self.failed_results),
                    'results': [r.to_dict() for r in self.failed_results]
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 失败结果保存至: {failed_file}")

        # 保存永久失败结果
        if self.permanently_failed_results:
            perm_failed_file = os.path.join(output_dir, f"{prefix}permanently_failed_{timestamp}.json")
            with open(perm_failed_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'retry_attempts': self.retry_attempts,
                    'total': len(self.permanently_failed_results),
                    'results': [r.to_dict() for r in self.permanently_failed_results],
                    'note': f'这些任务已重试{self.retry_attempts}次后仍失败'
                }, f, ensure_ascii=False, indent=2)
            print(f"⛔ 永久失败结果保存至: {perm_failed_file}")

    def print_statistics(self, stats: Dict[str, Any]):
        """
        打印统计信息（统一的输出格式）

        Args:
            stats: 统计字典
        """
        print(f"\n{'='*80}")
        print(f"📊 {self.__class__.__name__} 处理统计")
        print(f"{'='*80}")
        print(f"总任务数: {stats['total']}")
        print(f"✅ 成功: {stats['successful']} | ❌ 失败: {stats['failed']} | 🌐 API失败: {stats['api_failed']}")
        print(f"⛔ 永久失败: {stats['permanently_failed']}")
        print(f"📈 成功率: {stats['success_rate']:.1f}%")
        print(f"⏱️  总耗时: {stats['execution_time_minutes']:.2f} 分钟")
        print(f"⏱️  平均每任务: {stats['avg_time_per_task']:.2f} 秒")
        print(f"{'='*80}")

    @abstractmethod
    def process_single_task(self, task_data: Any) -> ProcessResult:
        """
        处理单个任务（子类必须实现）

        Args:
            task_data: 任务数据

        Returns:
            处理结果
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        运行处理器（子类必须实现）

        Returns:
            统计信息
        """
        pass


class BaseWorkflow(ABC, LoggerMixin, TimerMixin):
    """
    工作流基类 - 抽象工作流的公共逻辑

    所有工作流组件（ScenarioGenerator, CaseGenerator等）都继承此类
    混入LoggerMixin和TimerMixin提供日志记录和计时功能
    """

    def __init__(self, llm, output_dir: Optional[str] = None):
        """
        初始化工作流组件

        Args:
            llm: LLM模型实例
            output_dir: 输出目录
        """
        TimerMixin.__init__(self)  # 初始化TimerMixin
        self.llm = llm
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def save_json(self, data: Any, filename: str):
        """
        保存JSON文件（统一的保存逻辑）

        Args:
            data: 要保存的数据
            filename: 文件名
        """
        if self.output_dir is None:
            raise ValueError("output_dir 未设置")

        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return file_path

    def load_json(self, filename: str) -> Any:
        """
        加载JSON文件（统一的加载逻辑）

        Args:
            filename: 文件名

        Returns:
            加载的数据
        """
        if self.output_dir is None:
            raise ValueError("output_dir 未设置")

        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        执行工作流步骤（子类必须实现）

        Returns:
            执行结果
        """
        pass





