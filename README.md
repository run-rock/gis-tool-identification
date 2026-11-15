# GIS Tool Identification

This repository accompanies a study on GIS tool identification. It contains the implementation of embedding-based retrieval and TTPE tool selection, the evaluation datasets, and the result summaries for the manuscript. It also includes scripts for synthesizing and preprocessing training examples for GIS tool retrieval.

## Repository layout

```text
.
├── README.md
├── environment.yml
├── core/
│   ├── agent/                     # Candidate retrieval and TTPE tool selection
│   ├── common/                    # Embedding model interface
│   ├── preprocessing/             # Training sample preparation
│   └── synthesis/                 # GIS scenario and case synthesis
├── scripts/
│   ├── preprocessing/             # Dataset splitting and triplet preparation
│   └── synthesis/                 # Entry points for case synthesis
├── evaluation/
│   ├── embedding_eval/
│   │   ├── build_all_vectordbs.py
│   │   ├── evaluate_retrieval.py
│   │   └── evaluator.py
│   └── agent_eval/
│       ├── evaluate_toolbox_tool_matcher.py
│       ├── 4_evaluate_with_direct_llm_identification.py
│       └── 4_evaluate_with_direct_llm_不提供任何工具信息.py
├── data/
│   ├── qgis_tool/                 # Tool descriptions, mappings, and tool list
│   └── retrival_G8/测试数据/
│       ├── 工具检索和识别/
│       │   ├── tool_multi_answer.json
│       │   └── v1-6-人工校验v2.json
│       ├── 工具箱检索/
│       │   └── toolbox_multi_answer.json
│       └── 案例检索和选择/
│           └── case_from_tool_multi.json
└── results/
    ├── embedding/                 # Nine original CSV summaries for Table 4
    ├── ttpe/summary.csv
    ├── llm_comparison/summary.csv
    └── information_source/summary.csv
```

## Manuscript experiments

Dataset paths below are relative to `data/retrival_G8/测试数据/`.

| Experiment | Evaluation data | Results |
| --- | --- | --- |
| **Embedding retrieval (Table 4)** | Tool-Eval uses `工具检索和识别/tool_multi_answer.json` (3,642 queries). Toolbox-Eval uses `工具箱检索/toolbox_multi_answer.json` (3,642 queries). Case-Eval uses `案例检索和选择/case_from_tool_multi.json` (382 queries). | [`results/embedding/`](results/embedding/) |
| **Embedding Only vs. Embedding + TTPE** | `工具检索和识别/v1-6-人工校验v2.json` (100 manually reviewed queries) | [`results/ttpe/summary.csv`](results/ttpe/summary.csv) |
| **LLM accuracy and runtime comparison** | `工具检索和识别/v1-6-人工校验v2.json` (100 manually reviewed queries) | [`results/llm_comparison/summary.csv`](results/llm_comparison/summary.csv) |
| **Information-source comparison** | The reported accuracy and error counts reconcile with the 3,642-query full tool test set, `工具检索和识别/tool_multi_answer.json`. The available Kimi timing logs use the 100-query manually reviewed set. The original full-set prediction files and the provenance of the table’s timing values remain to be confirmed. | [`results/information_source/summary.csv`](results/information_source/summary.csv) |

For Table 4, the source CSV fields are `recall@1`, `recall@3`, and `recall@5` for the manuscript’s P@k columns, and `ndcg@1`, `ndcg@3`, and `ndcg@5` for N@k. The three comparison CSVs distinguish manuscript values from values in the available evaluation files wherever they differ.

In the information-source comparison, the reported error counts of 110, 328, and 2,331 yield accuracies of 96.98%, 90.99%, and 36.00%, respectively, when divided by the full test set size using `accuracy = 1 − error_count / 3642`. This establishes the denominator implied by the table. The archived 100-query runs provide separate evaluation records.

## Running the evaluations

Create and activate the environment from the repository root.

```bash
conda env create -f environment.yml
conda activate gis-tool-identification
```

### Embedding retrieval

Use `evaluation/embedding_eval/evaluate_retrieval.py` with the test dataset, the embedding model configuration, and the matching FAISS index prefix for the tool, toolbox, or case retrieval task. Each index consists of a `.faiss` file and its corresponding `.pkl` file. The index-building entry point is `evaluation/embedding_eval/build_all_vectordbs.py`.

```bash
python evaluation/embedding_eval/evaluate_retrieval.py --help
python evaluation/embedding_eval/build_all_vectordbs.py --help
```

### TTPE and the LLM comparison

Use `evaluation/agent_eval/evaluate_toolbox_tool_matcher.py` with the tool and toolbox indexes, the tool-to-toolbox mapping, the test file, and the LLM provider and model settings. The identified Kimi comparison run used three candidate toolboxes and three candidate tools.

```bash
python evaluation/agent_eval/evaluate_toolbox_tool_matcher.py --help
```

### Information-source controls

The two `4_evaluate_with_direct_llm` scripts in `evaluation/agent_eval/` implement the direct tool-selection controls. **Tools Information** provides the LLM with [`all_tools_without_params.txt`](data/qgis_tool/all_tools_without_params.txt). **Nothing Information** provides the task description alone. Both scripts accept an evaluation dataset as input.

## Training and model artifacts

The synthesis entry points are in `scripts/synthesis/`, and the dataset splitting and triplet preparation entry points are in `scripts/preprocessing/`. Synthesis requires an OpenAI-compatible model service and local GIS tool descriptions.

The training codebase was subsequently refactored. Its training entry point is `scripts/training/run_train.py` in the source project. That entry point and its supporting modules are outside the current repository snapshot.

The training data, LoRA adapter, and prebuilt Qwen3-Embedding-8B FAISS indexes are packaged separately. The index-building examples reference the LoRA adapter from `checkpoint-3200`. The TTPE run records identify the `v1` tool and toolbox indexes, while the exact checkpoint used to build those archived indexes remains unconfirmed. Download links will be added when the separate artifact archive is published.
