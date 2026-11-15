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
