# Evaluation

We use VLMEVALKIT as the evaluation tools. Please refer to [QuickStart](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) for installation.

We evaluate the models with scripts in [evaluation.sh](scripts/evaluation.sh). You could modify the parameters for evaluating different benchmarks.

```bash
eval_dataset="MMVet" # need openai api key
eval_dataset="MME MMBench_DEV_EN"
```