# GIS Tool Identification

This repository contains GIS tool identification, embedding retrieval evaluation and TTPE. Run scripts from the repository root. Set model, index and service paths explicitly. The synthesis scripts require an OpenAI-compatible service and local GIS tool descriptions.

| Experiment | Evaluation entry point | Dataset relative to data/retrival_G8/测试数据 | Results |
|---|---|---|---|
| Tool-Eval | evaluation/embedding_eval/evaluate_retrieval.py | 工具检索和识别/tool_multi_answer.json | results/embedding |
| Toolbox-Eval | evaluation/embedding_eval/evaluate_retrieval.py | 工具箱检索/toolbox_multi_answer.json | results/embedding |
| Case-Eval | evaluation/embedding_eval/evaluate_retrieval.py | 案例检索和选择/case_from_tool_multi.json | results/embedding |
| TTPE and LLM comparison | evaluation/agent_eval/evaluate_toolbox_tool_matcher.py | 工具检索和识别/v1-6-人工校验v2.json | results/ttpe and results/llm_comparison |
| Information source | TTPE plus the two evaluation/agent_eval/4_evaluate_with_direct_llm scripts | 工具检索和识别/v1-6-人工校验v2.json | results/information_source |

Tools Information additionally uses data/qgis_tool/all_tools_without_params.txt. The historical 8B training entry point is still being recovered. Checkpoint 3200 is referenced by the index-building examples, and the exact provenance of the published v1 index remains to be verified. No evaluation was rerun for this draft.

## Repository layout

```text
.
├── core/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── data_parser.py
│   │   ├── toolbox_tool_matcher_3.py
│   │   └── vector_db.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── embedding_model.py
│   ├── preprocessing/
│   │   ├── builders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── case_tool.py
│   │   │   ├── tool_case.py
│   │   │   ├── tool_toolbox.py
│   │   │   ├── toolbox_case.py
│   │   │   └── toolbox_tool.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── converter.py
│   │   ├── formatters.py
│   │   └── splitter.py
│   ├── synthesis/
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── stage1_batch_processor.py
│   │   │   ├── stage1_result_batch_processor.py
│   │   │   └── stage2_batch_processor.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── embedding_service.py
│   │   │   ├── llm_utils.py
│   │   │   └── vector_db.py
│   │   ├── workflows/
│   │   │   ├── __init__.py
│   │   │   ├── case_generator.py
│   │   │   ├── case_reviser.py
│   │   │   ├── case_validator.py
│   │   │   ├── scenario_generator.py
│   │   │   ├── stage1_orchestrator.py
│   │   │   ├── stage1_result_extractor.py
│   │   │   └── stage2_orchestrator.py
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── models.py
│   │   └── prompts.py
│   └── __init__.py
├── data/
│   ├── qgis_tool/
│   │   ├── all_tools_without_params.txt
│   │   ├── qgis_tool_with_paramaters重写工具集描述-字符串版.json
│   │   └── qgis_tool_with_paramaters重写工具集描述.json
│   └── retrival_G8/
│       └── 测试数据/
│           ├── 工具检索和识别/
│           │   ├── tool_multi_answer.json
│           │   └── v1-6-人工校验v2.json
│           ├── 工具箱检索/
│           │   └── toolbox_multi_answer.json
│           └── 案例检索和选择/
│               └── case_from_tool_multi.json
├── evaluation/
│   ├── agent_eval/
│   │   ├── 4_evaluate_with_direct_llm_identification.py
│   │   ├── 4_evaluate_with_direct_llm_不提供任何工具信息.py
│   │   ├── __init__.py
│   │   └── evaluate_toolbox_tool_matcher.py
│   ├── embedding_eval/
│   │   ├── __init__.py
│   │   ├── build_all_vectordbs.py
│   │   ├── evaluate_retrieval.py
│   │   └── evaluator.py
│   └── __init__.py
├── results/
│   ├── embedding/
│   │   ├── Qwen3-Embedding-8b/
│   │   │   ├── base/
│   │   │   │   ├── 工具/
│   │   │   │   │   └── evaluation_summary.csv
│   │   │   │   ├── 工具箱/
│   │   │   │   │   └── evaluation_summary.csv
│   │   │   │   └── 案例/
│   │   │   │       └── evaluation_summary.csv
│   │   │   └── v1/
│   │   │       ├── 工具/
│   │   │       │   └── evaluation_summary.csv
│   │   │       ├── 工具集/
│   │   │       │   └── evaluation_summary.csv
│   │   │       └── 案例/
│   │   │           └── evaluation_summary.csv
│   │   └── jina/
│   │       ├── 工具/
│   │       │   └── evaluation_summary.csv
│   │       ├── 工具箱/
│   │       │   └── evaluation_summary.csv
│   │       └── 案例/
│   │           └── evaluation_summary.csv
│   ├── information_source/
│   │   └── summary.csv
│   ├── llm_comparison/
│   │   └── summary.csv
│   └── ttpe/
│       └── summary.csv
├── scripts/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── run_preprocess_build.py
│   │   ├── run_preprocess_convert.py
│   │   └── run_preprocess_split.py
│   └── synthesis/
│       ├── __init__.py
│       ├── run_stage1.py
│       ├── run_stage1_extract.py
│       └── run_stage2.py
├── README.md
└── environment.yml
```


## Evaluation workflow

- Install the dependencies with `conda env create -f environment.yml` and activate `gis-tool-identification`.
- Run `python evaluation/embedding_eval/evaluate_retrieval.py --help` for embedding retrieval. Set the task-specific test file and FAISS index prefix.
- Run `python evaluation/agent_eval/evaluate_toolbox_tool_matcher.py --help` for TTPE. The identified Kimi runs used three candidate toolboxes and three candidate tools.
- For the information-source controls, use the two direct-LLM evaluators under `evaluation/agent_eval/`. Both use the manually reviewed tool-identification test set. The Tools Information control also reads `data/qgis_tool/all_tools_without_params.txt`.

The embedding results directory contains the nine original summary CSVs. The other three result directories contain comparison CSVs that preserve values from both the manuscript and the available logs. The manuscript's false-retrieval counts remain unverified.

## Model and index artifacts

The Qwen3-Embedding-8B base model, candidate LoRA adapter from checkpoint 3200, and the FAISS indexes are managed separately. Each index needs its matching `.faiss` and `.pkl` files. Use the tool, toolbox, or case index for the corresponding retrieval task. Archive links are pending.
