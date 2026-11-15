# GIS Tool Identification

This repository provides the code, evaluation datasets, and result summaries for the GIS tool identification experiments reported in the paper. The experiments cover embedding-based retrieval, TTPE tool selection from candidate tools and toolboxes, the generalization of TTPE across large language models (LLMs), and two baselines in which an LLM selects a tool directly. The repository also includes code for synthesizing and preprocessing training samples for GIS tool identification.

## Repository Structure

```text
.
├── README.md
├── environment.yml
├── core/
│   ├── agent/                     # Candidate retrieval, vector index interface, and TTPE tool selection
│   ├── common/                    # Embedding model interface
│   ├── preprocessing/             # Training sample construction and format conversion
│   ├── synthesis/                 # GIS scenario and case synthesis
│   └── training/
│       ├── __init__.py
│       ├── config.py              # Model, data paths, and training settings
│       ├── config_logger.py       # Training configuration logging
│       ├── data_loader.py         # Training and validation data loading
│       ├── distance_analyzer.py   # Embedding distance analysis
│       ├── losses.py              # Triplet and contrastive losses
│       └── utils.py               # Pooling, batch collation, and training callbacks
├── scripts/
│   ├── preprocessing/            # Dataset splitting, triplet construction, and format conversion
│   ├── synthesis/                 # Scenario and case synthesis entry points
│   └── training/
│       └── run_train.py           # Embedding model training entry point
├── evaluation/
│   ├── embedding_eval/
│   │   ├── build_all_vectordbs.py  # Build tool, toolbox, and case indexes
│   │   ├── evaluate_retrieval.py   # Embedding retrieval evaluation
│   │   └── evaluator.py           # Retrieval metric computation
│   └── agent_eval/
│       ├── evaluate_toolbox_tool_matcher.py              # TTPE evaluation and cross-LLM comparison
│       ├── 4_evaluate_with_direct_llm_identification.py   # Baseline with tool information
│       └── 4_evaluate_with_direct_llm_不提供任何工具信息.py  # Baseline with the task description alone
├── data/
│   ├── qgis_tool/
│   │   ├── qgis_tool_with_paramaters重写工具集描述.json
│   │   ├── qgis_tool_with_paramaters重写工具集描述-字符串版.json
│   │   └── all_tools_without_params.txt                  # Tool list for the Tools Information baseline
│   └── retrival_G8/测试数据/
│       ├── 工具检索和识别/
│       │   ├── tool_multi_answer.json                    # Full tool retrieval test set (3,642 queries)
│       │   └── v1-6-人工校验v2.json                        # Manually verified tool identification set (100 tasks)
│       ├── 工具箱检索/
│       │   └── toolbox_multi_answer.json                 # Toolbox retrieval test set (3,642 queries)
│       └── 案例检索和选择/
│           └── case_from_tool_multi.json                 # Case retrieval test set (382 queries)
└── results/
    ├── embedding/                 # Nine original CSV summaries for embedding retrieval
    ├── ttpe/summary.csv           # Embedding Only vs. Embedding + TTPE
    ├── llm_comparison/summary.csv  # TTPE generalization across LLMs
    └── information_source/summary.csv  # Comparison of three information sources
```

## Experiments and Datasets

Dataset paths in the table are relative to `data/retrival_G8/测试数据/`.

| Experiment | Test dataset | Results |
| --- | --- | --- |
| **Embedding retrieval** | Tool retrieval uses `工具检索和识别/tool_multi_answer.json` (3,642 queries). Toolbox retrieval uses `工具箱检索/toolbox_multi_answer.json` (3,642 queries). Case retrieval uses `案例检索和选择/case_from_tool_multi.json` (382 queries). | [`results/embedding/`](results/embedding/) |
| **Embedding Only vs. Embedding + TTPE** | `工具检索和识别/v1-6-人工校验v2.json` (100 manually verified tasks) | [`results/ttpe/summary.csv`](results/ttpe/summary.csv) |
| **Accuracy and runtime comparison across LLMs** | `工具检索和识别/v1-6-人工校验v2.json` (100 manually verified tasks) | [`results/llm_comparison/summary.csv`](results/llm_comparison/summary.csv) |
| **Information-source comparison** | `工具检索和识别/v1-6-人工校验v2.json` (100 manually verified tasks) | [`results/information_source/summary.csv`](results/information_source/summary.csv) |

## Running the Evaluations

Create and activate the environment from the repository root.

```bash
conda env create -f environment.yml
conda activate gis-tool-identification
```

### Embedding Retrieval

Run `evaluation/embedding_eval/evaluate_retrieval.py` with the test dataset, embedding model configuration, and FAISS index path prefix for the task. Tool, toolbox, and case retrieval each use a separate index. Each index requires a matching pair of `.faiss` and `.pkl` files.

```bash
python evaluation/embedding_eval/evaluate_retrieval.py --help
python evaluation/embedding_eval/build_all_vectordbs.py --help
```

### TTPE and Cross-LLM Comparison

Run `evaluation/agent_eval/evaluate_toolbox_tool_matcher.py` with the test dataset, tool and toolbox indexes, tool-to-toolbox mapping file, and LLM provider and model settings. The Kimi comparison run used **three candidate toolboxes and three candidate tools**.

```bash
python evaluation/agent_eval/evaluate_toolbox_tool_matcher.py --help
```

### Information-Source Comparison

- **Embedding Model** retrieves candidate tools and toolboxes using embeddings, then applies TTPE to select a tool.
- **Tools Information** provides the LLM with the task description and a [list of tools](data/qgis_tool/all_tools_without_params.txt). The LLM selects a tool directly.
- **Nothing Information** provides the LLM with the task description alone and asks it to predict the tool.

The two direct-selection evaluation scripts are in `evaluation/agent_eval/`. Specify the test dataset when running either script.

## Training and Model Files

Training sample synthesis entry points are in `scripts/synthesis/`. Dataset splitting, triplet construction, and format conversion entry points are in `scripts/preprocessing/`.

The embedding model training entry point is `scripts/training/run_train.py`.

The training data, LoRA weights, and prebuilt Qwen3-Embedding-8B FAISS indexes are packaged separately. Download links will be added when the archives are released. The index-building examples use the LoRA weights from `checkpoint-3200`. The TTPE runs use the `v1` tool and toolbox indexes, which were built from `checkpoint-3200`.
