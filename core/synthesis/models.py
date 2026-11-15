#!/usr/bin/env python3
"""
数据模型定义 - 统一的数据结构
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class CaseStatus(Enum):
    """案例状态枚举"""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    REVISED = "revised"
    FAILED = "failed"


class ProcessStatus(Enum):
    """处理状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    RUNNING = "running"


# 看到 @dataclass，你就把它理解为：“Python，请帮我自动把这个类变成一个标准的数据容器，把那些枯燥的初始化和打印代码都帮我写好。”
@dataclass
class Tool:
    """工具数据模型"""
    name: str
    algorithm_id: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    '''
        这是一个非常重要且常用的装饰器。

    简单来说，@classmethod 用来定义**“类方法”**。

    它的核心特征是：这个方法不需要实例化（创建对象）就能调用，而且它收到的第一个参数不是 self（实例本身），而是 cls（类本身）。
    1. 为什么需要它？（核心场景：工厂模式）

    在你的代码中，from_dict 就是最经典的使用场景：“替代构造函数”。

    通常我们创建对象是用类名加括号：
    code Python

        
    t = Tool("锤子", "001", "敲东西") # 调用默认的 __init__

    

    但有时候，你的数据来源是一个字典（比如从 JSON 读出来的）。如果不用 @classmethod，你得在外部手动解析字典再传进去。

    用了 @classmethod，你可以把这个逻辑封装在类里面：
    code Python

        
    data = {
        "name": "扳手",
        "algorithm_id": "002",
        "description": "拧螺丝",
        "parameters": {"size": 10}
    }

    # 直接通过类名调用，不需要先创建一个对象
    # 这就像一个“工厂”，输入字典，产出一个 Tool 对象
    my_tool = Tool.from_dict(data)

    

    2. 语法对比：普通方法 vs 类方法
    特性	普通方法 (Instance Method)	类方法 (@classmethod)
    装饰器	无	@classmethod
    第一个参数	self (代表具体的对象实例)	cls (代表类 Tool 本身)
    调用方式	必须先创建对象：obj.method()	直接用类名：Class.method()
    用途	操作具体的对象数据（如 to_dict）	创建对象、修改全局配置
    3. 代码逐行解读

    让我们看你提供的代码片段：
    code Python

        
    @classmethod
        def from_dict(cls, data: Dict) -> 'Tool':
            """从字典创建"""
            # 这里的 cls 就等于 Tool 这个类
            # cls(...) 就等于 Tool(...)，也就是在调用构造函数
            return cls(
                name=data['name'],
                algorithm_id=data['algorithm_id'],
                description=data['description'],
                parameters=data.get('parameters', {})
            )

  
    '''
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Tool':
        """从字典创建"""
        return cls(
            name=data['name'],
            algorithm_id=data['algorithm_id'],
            description=data['description'],
            parameters=data.get('parameters', {})
        )


@dataclass
class Toolbox:
    """工具箱数据模型"""
    name: str
    description: str

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Toolbox':
        """从字典创建"""
        return cls(
            name=data.get('toolbox', data.get('name', '')),
            description=data.get('discription', data.get('description', ''))
        )


@dataclass
class Scenario:
    """场景数据模型"""
    name: str
    description: str
    scenario_id: Optional[int] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Scenario':
        """从字典创建"""
        return cls(
            name=data['name'],
            description=data['description'],
            scenario_id=data.get('scenario_id')
        )


@dataclass
class Case:
    """案例数据模型"""
    case_id: int
    scenario_name: str
    generated_case: str
    status: CaseStatus = CaseStatus.PENDING
    check_result: Optional[Dict] = None
    revision_result: Optional[Dict] = None
    similar_tools: List[Dict] = field(default_factory=list)
    evaluation: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Case':
        """从字典创建"""
        status_str = data.get('status', 'pending')
        status = CaseStatus(status_str) if isinstance(status_str, str) else status_str

        return cls(
            case_id=data['case_id'],
            scenario_name=data['scenario_name'],
            generated_case=data['generated_case'],
            status=status,
            check_result=data.get('check_result'),
            revision_result=data.get('revision_result'),
            similar_tools=data.get('similar_tools', []),
            evaluation=data.get('evaluation'),
            metadata=data.get('metadata', {})
        )


@dataclass
class WorkflowResult:
    """工作流结果数据模型"""
    tool: Tool
    toolbox: Toolbox
    scenarios: List[Scenario]
    cases: List[Case]
    statistics: Dict[str, Any]
    status: ProcessStatus = ProcessStatus.SUCCESS
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'tool': self.tool.to_dict(),
            'toolbox': self.toolbox.to_dict(),
            'scenarios': [s.to_dict() for s in self.scenarios],
            'cases': [c.to_dict() for c in self.cases],
            'statistics': self.statistics,
            'status': self.status.value,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkflowResult':
        """从字典创建"""
        return cls(
            tool=Tool.from_dict(data['tool']),
            toolbox=Toolbox.from_dict(data['toolbox']),
            scenarios=[Scenario.from_dict(s) for s in data.get('scenarios', [])],
            cases=[Case.from_dict(c) for c in data.get('cases', [])],
            statistics=data.get('statistics', {}),
            status=ProcessStatus(data.get('status', 'success')),
            error_message=data.get('error_message')
        )

    def get_valid_cases(self) -> List[Case]:
        """获取所有有效案例"""
        return [c for c in self.cases if c.status == CaseStatus.VALID]

    def get_invalid_cases(self) -> List[Case]:
        """获取所有无效案例"""
        return [c for c in self.cases if c.status == CaseStatus.INVALID]


@dataclass
class ProcessResult:
    """处理结果数据模型（用于并行处理）"""
    status: ProcessStatus
    tool_id: str
    tool_name: str
    toolbox_name: str
    duration_minutes: float
    log_file: str
    result: Optional[Any] = None
    error: Optional[str] = None
    is_api_error: bool = False
    retry_count: int = 0

    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessResult':
        """从字典创建"""
        status_str = data.get('status', 'success')
        status = ProcessStatus(status_str) if isinstance(status_str, str) else status_str

        return cls(
            status=status,
            tool_id=data['tool_id'],
            tool_name=data['tool_name'],
            toolbox_name=data['toolbox_name'],
            duration_minutes=data.get('duration_minutes', 0.0),
            log_file=data.get('log_file', ''),
            result=data.get('result'),
            error=data.get('error'),
            is_api_error=data.get('is_api_error', False),
            retry_count=data.get('retry_count', 0)
        )
