#!/usr/bin/env python3
"""
QGIS 案例合成系统 - 核心模块

目录结构：
- base.py, config.py, models.py, prompts.py: 基础层
- services/: 可复用服务组件（LLM工具、Embedding服务、向量数据库）
- workflows/: 工作流子组件（Stage1、中间结果提取、Stage2）
- processors/: 批处理器（并行处理多个工具）
- scripts/: CLI入口脚本

注意：使用懒加载模式，只在需要时导入具体模块，避免循环依赖
"""

# 不在包初始化时导入所有内容，让子模块按需导入
# 这样可以避免循环依赖和导入顺序问题
