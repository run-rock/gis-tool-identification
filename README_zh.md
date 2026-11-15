
# GIS 工具识别

本仓库提供论文中 GIS 工具识别实验的代码、测试数据和结果汇总。实验包括嵌入模型（Embedding）检索、基于候选工具与工具箱的 TTPE 工具选择、TTPE方法的llm泛化实验，以及直接使用大语言模型（LLM）选择工具的对照。仓库也包含用于合成和预处理工具识别训练样本的代码。

## 文件结构

```text
.
├── README.md
├── environment.yml
├── core/
│   ├── agent/                     # 候选检索、向量库接口和 TTPE 工具选择
│   ├── common/                    # Embedding 模型接口
│   ├── preprocessing/            # 训练样本构建与格式转换
│   └── synthesis/                 # GIS 场景与案例合成
│   └── training/
│       ├── __init__.py
│       ├── config.py              # 模型、数据路径和训练配置
│       ├── config_logger.py       # 记录训练配置
│       ├── data_loader.py         # 加载训练和验证样本
│       ├── distance_analyzer.py   # 分析样本的向量距离
│       ├── losses.py              # 三元组损失和对比损失
│       └── utils.py               # 池化、批次整理和训练回调
├── scripts/
│   ├── preprocessing/            # 数据划分、三元组构造和格式转换入口
│   └── synthesis/                 # 场景与案例合成入口
│   └── training/
│       └── run_train.py           # Embedding 模型训练入口
├── evaluation/
│   ├── embedding_eval/
│   │   ├── build_all_vectordbs.py  # 构建工具、工具箱和案例向量库
│   │   ├── evaluate_retrieval.py   # Embedding 检索评估入口
│   │   └── evaluator.py           # 检索指标计算
│   └── agent_eval/
│       ├── evaluate_toolbox_tool_matcher.py              # TTPE 主实验与 LLM 泛化实验
│       ├── 4_evaluate_with_direct_llm_identification.py    # 提供工具信息的对照
│       └── 4_evaluate_with_direct_llm_不提供任何工具信息.py  # 仅提供任务描述的对照
├── data/
│   ├── qgis_tool/
│   │   ├── qgis_tool_with_paramaters重写工具集描述.json
│   │   ├── qgis_tool_with_paramaters重写工具集描述-字符串版.json
│   │   └── all_tools_without_params.txt                  # Tools Information 对照的工具清单
│   └── retrival_G8/测试数据/
│       ├── 工具检索和识别/
│       │   ├── tool_multi_answer.json                   # 全量工具检索测试集（3,642 条）
│       │   └── v1-6-人工校验v2.json                       # 人工校验工具识别测试集（100 条）
│       ├── 工具箱检索/
│       │   └── toolbox_multi_answer.json                # 工具箱检索测试集（3,642 条）
│       └── 案例检索和选择/
│           └── case_from_tool_multi.json                # 案例检索测试集（382 条）
└── results/
    ├── embedding/                 # Embedding 检索实验的九份原始汇总 CSV
    ├── ttpe/summary.csv           # Embedding Only 与 Embedding + TTPE 对照
    ├── llm_comparison/summary.csv  # TTPE 的 LLM 泛化实验
    └── information_source/summary.csv  # 三种信息来源对照实验
```

## 论文实验与数据

下表的测试集路径均相对于 `data/retrival_G8/测试数据/`。

| 实验 | 测试集 | 结果文件 |
| --- | --- | --- |
| **Embedding 检索** | 工具检索用 `工具检索和识别/tool_multi_answer.json`（3,642 条）；工具箱检索用 `工具箱检索/toolbox_multi_answer.json`（3,642 条）；案例检索用 `案例检索和选择/case_from_tool_multi.json`（382 条） | [`results/embedding/`](results/embedding/) |
| **Embedding Only 与 Embedding + TTPE 对照** | `工具检索和识别/v1-6-人工校验v2.json`（100 条人工校验任务） | [`results/ttpe/summary.csv`](results/ttpe/summary.csv) |
| **不同 LLM 的准确率与耗时对比** | `工具检索和识别/v1-6-人工校验v2.json`（100 条人工校验任务） | [`results/llm_comparison/summary.csv`](results/llm_comparison/summary.csv) |
| **不同信息来源对照** | `工具检索和识别/v1-6-人工校验v2.json`（100 条人工校验任务）| [`results/information_source/summary.csv`](results/information_source/summary.csv) |



## 运行评估

在仓库根目录创建并激活环境：

```bash
conda env create -f environment.yml
conda activate gis-tool-identification
```

### Embedding 检索

使用 `evaluation/embedding_eval/evaluate_retrieval.py`，按评估任务指定测试集、模型配置和 FAISS 索引路径前缀。工具检索、工具箱检索和案例检索分别使用对应的索引。每套索引的 `.faiss` 和 `.pkl` 文件需要配套使用。

```bash
python evaluation/embedding_eval/evaluate_retrieval.py --help
python evaluation/embedding_eval/build_all_vectordbs.py --help
```

### TTPE 与 LLM 对比

使用 `evaluation/agent_eval/evaluate_toolbox_tool_matcher.py`，指定测试集、工具索引、工具箱索引、工具与工具箱的映射文件，以及 LLM 服务和模型。已定位的 Kimi 对照运行使用了 **3 个候选工具箱和 3 个候选工具**。

```bash
python evaluation/agent_eval/evaluate_toolbox_tool_matcher.py --help
```

### 信息来源对照实验

- **Embedding Model** 使用 Embedding 检索候选工具和工具箱，再通过 TTPE 选择工具。
- **Tools Information** 将任务描述和[工具清单](data/qgis_tool/all_tools_without_params.txt)提供给 LLM，由它直接选择工具。
- **Nothing Information** 只向 LLM 提供任务描述，由它直接预测工具。

后两种对照的评估脚本位于 `evaluation/agent_eval/`，运行时通过参数指定测试集。

## 训练与模型文件

训练样本合成入口位于 `scripts/synthesis/`。数据划分、三元组构造和格式转换入口位于 `scripts/preprocessing/`。

Embedding 训练入口是 `scripts/training/run_train.py`。

训练数据、LoRA 权重和预构建的 Qwen3-Embedding-8B FAISS 索引已分别打包，归档下载链接待发布。建库示例引用 `checkpoint-3200` 的 LoRA 权重，TTPE 运行记录使用 `v1` 的工具和工具箱索引。归档中这批 `v1` 索引由`checkpoint-3200`构建。
