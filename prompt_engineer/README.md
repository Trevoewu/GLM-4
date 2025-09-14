## CMCC Prompt Evaluation (ChatGLM4-9B)

This KISS-style utility evaluates three prompt templates (Zero-shot, Few-shot, CoT) on the CMCC test split and reports Accuracy and Macro F1 for `办理/投诉/咨询`.

### Requirements
- Conda env: `glm4` 
- Local GLM4-9B server running at `http://127.0.0.1:8001` with OpenAI-compatible `/v1/chat/completions`
- Dataset at `/data/glm4/data/cmcc-34/test_new.csv`

### Run

```bash
conda activate glm4
python prompt_engineer/eval_cmcc_prompts.py
```

Optional environment variables:
- `GLM4_BASE_URL` (default `http://127.0.0.1:8001`)
- `GLM4_MODEL_NAME` (default `glm4-9b`)
- `CMCC_TEST_PATH` (default `/data/glm4/data/cmcc-34/test_new.csv`)
- `RESULTS_WEBHOOK_URL` (POSTs JSON results if set)

Results will be saved to `prompt_engineer/output/prompt_eval_results.json` and printed to stdout.

