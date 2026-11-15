#!/usr/bin/env python3
"""
第一阶段结果批量处理器

批量处理多个工具的第一阶段结果，提取最终有效案例
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from ..workflows.stage1_result_extractor import Stage1ResultExtractor


class Stage1ResultBatchProcessor:
    """
    第一阶段结果批量处理器

    负责批量处理第一阶段的工作流结果：
    1. 查找所有工具目录
    2. 并行提取每个工具的最终有效案例
    3. 生成统计报告
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        max_workers: int = 10
    ):
        """
        初始化批量处理器

        Args:
            input_dir: 输入目录（第一阶段输出目录，包含各工具子目录）
            output_dir: 输出目录（保存最终案例文件）
            max_workers: 最大并发线程数
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 第一阶段结果批量处理器初始化")
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🧵 最大线程数: {max_workers}")

    def _find_tool_directories(self) -> List[str]:
        """
        查找所有工具目录

        Returns:
            工具目录路径列表
        """
        tool_dirs = []

        if not os.path.exists(self.input_dir):
            print(f"❌ 输入目录不存在: {self.input_dir}")
            return tool_dirs

        for item in os.listdir(self.input_dir):
            item_path = os.path.join(self.input_dir, item)

            # 检查是否为目录
            if not os.path.isdir(item_path):
                continue

            # 检查目录中是否有工作流结果文件
            files = os.listdir(item_path)
            has_check_result = any(
                f.startswith('check_result_') and f.endswith('.json')
                for f in files
            )

            if has_check_result:
                tool_dirs.append(item_path)
                print(f"📁 找到工具目录: {item}")
            else:
                print(f"⚠️  跳过目录 {item}（缺少工作流结果文件）")

        return tool_dirs

    def _process_single_tool(self, tool_dir: str) -> Dict[str, Any]:
        """
        处理单个工具目录

        Args:
            tool_dir: 工具目录路径

        Returns:
            处理结果字典
        """
        tool_name = os.path.basename(tool_dir)

        try:
            # 创建提取器
            extractor = Stage1ResultExtractor()

            # 提取案例
            result = extractor.extract_from_directory(
                tool_workflow_dir=tool_dir,
                output_dir=self.output_dir
            )

            return {
                'status': 'success',
                'tool_name': tool_name,
                'tool_dir': tool_dir,
                'output_file': result['output_file'],
                'total_cases': result['total_cases'],
                'initial_valid_cases': result['initial_valid_cases'],
                'final_valid_cases': result['final_valid_cases']
            }

        except Exception as e:
            return {
                'status': 'failed',
                'tool_name': tool_name,
                'tool_dir': tool_dir,
                'error': str(e)
            }

    def run(self) -> Dict[str, Any]:
        """
        运行批量处理

        Returns:
            统计结果
        """
        print(f"\n{'='*80}")
        print("🚀 开始批量提取最终案例")
        print(f"{'='*80}\n")

        # 查找所有工具目录
        tool_dirs = self._find_tool_directories()

        if not tool_dirs:
            print("❌ 未找到任何包含工作流结果的工具目录")
            return {
                'total_tools': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'execution_time': 0
            }

        total_tools = len(tool_dirs)
        print(f"\n📋 找到 {total_tools} 个工具目录需要处理")
        print(f"🧵 并行线程数: {self.max_workers}\n")

        # 记录开始时间
        start_time = datetime.now()

        # 使用ThreadPoolExecutor并行处理
        successful_results = []
        failed_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self._process_single_tool, tool_dir): tool_dir
                for tool_dir in tool_dirs
            }

            print(f"📤 已提交 {len(futures)} 个任务到线程池")
            print(f"🚀 开始处理...\n")

            # 处理完成的任务（带进度条）
            with tqdm(total=total_tools, desc="处理工具", ncols=100) as pbar:
                for future in as_completed(futures):
                    tool_dir = futures[future]
                    tool_name = os.path.basename(tool_dir)

                    try:
                        result = future.result()

                        if result['status'] == 'success':
                            successful_results.append(result)
                            pbar.set_postfix_str(f"✅ {tool_name}")
                        else:
                            failed_results.append(result)
                            pbar.set_postfix_str(f"❌ {tool_name}")

                    except Exception as e:
                        # 线程执行异常
                        failed_results.append({
                            'status': 'failed',
                            'tool_name': tool_name,
                            'tool_dir': tool_dir,
                            'error': f"线程执行异常: {str(e)}"
                        })
                        pbar.set_postfix_str(f"💥 {tool_name}")

                    pbar.update(1)

        # 计算统计信息
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        stats = {
            'total_tools': total_tools,
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / total_tools * 100 if total_tools > 0 else 0,
            'execution_time': execution_time,
            'avg_time_per_tool': execution_time / total_tools if total_tools > 0 else 0
        }

        # 计算总案例数
        total_cases = sum(r['total_cases'] for r in successful_results)
        total_initial_valid = sum(r['initial_valid_cases'] for r in successful_results)
        total_final_valid = sum(r['final_valid_cases'] for r in successful_results)

        stats['total_cases'] = total_cases
        stats['total_initial_valid_cases'] = total_initial_valid
        stats['total_final_valid_cases'] = total_final_valid

        # 打印统计结果
        self._print_statistics(stats, successful_results, failed_results)

        # 保存失败结果
        if failed_results:
            self._save_failed_results(failed_results)

        # 保存成功的文件列表
        self._save_success_list(successful_results)

        return stats

    def _print_statistics(
        self,
        stats: Dict[str, Any],
        successful_results: List[Dict],
        failed_results: List[Dict]
    ):
        """
        打印统计信息

        Args:
            stats: 统计字典
            successful_results: 成功结果列表
            failed_results: 失败结果列表
        """
        print(f"\n{'='*80}")
        print("📊 批量提取总结")
        print(f"{'='*80}")
        print(f"总工具数: {stats['total_tools']}")
        print(f"提取成功: {stats['successful']} ✅")
        print(f"提取失败: {stats['failed']} ❌")
        print(f"成功率: {stats['success_rate']:.1f}%")
        print(f"总执行时间: {stats['execution_time']:.1f}秒")
        print(f"平均每工具: {stats['avg_time_per_tool']:.1f}秒")

        if successful_results:
            print(f"\n📊 案例统计:")
            print(f"  总案例数: {stats['total_cases']}")
            print(f"  初始通过: {stats['total_initial_valid_cases']}")
            print(f"  修正通过: {stats['total_final_valid_cases']}")

        if failed_results:
            print(f"\n⚠️  失败的工具:")
            for result in failed_results[:10]:  # 只显示前10个
                print(f"  ❌ {result['tool_name']}: {result.get('error', '未知错误')}")
            if len(failed_results) > 10:
                print(f"  ... 还有 {len(failed_results) - 10} 个失败")

    def _save_failed_results(self, failed_results: List[Dict]):
        """
        保存失败结果

        Args:
            failed_results: 失败结果列表
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_file = os.path.join(self.output_dir, f"failed_tools_{timestamp}.json")

        failed_data = {
            'timestamp': timestamp,
            'total_failed': len(failed_results),
            'failed_tools': failed_results
        }

        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 失败工具记录保存至: {failed_file}")

    def _save_success_list(self, successful_results: List[Dict]):
        """
        保存成功的文件列表

        Args:
            successful_results: 成功结果列表
        """
        if not successful_results:
            return

        # 列出保存的文件
        print(f"\n📄 保存的文件列表:")
        saved_files = []

        for result in successful_results:
            file_name = os.path.basename(result['output_file'])
            saved_files.append({
                'file': file_name,
                'tool': result['tool_name'],
                'total_cases': result['total_cases']
            })
            print(f"   - {file_name} ({result['total_cases']} 案例)")

        # 保存文件列表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        list_file = os.path.join(self.output_dir, f"success_list_{timestamp}.json")

        with open(list_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_files': len(saved_files),
                'files': saved_files
            }, f, ensure_ascii=False, indent=2)

        print(f"\n💾 文件列表保存至: {list_file}")
