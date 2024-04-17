# Evaluation

We use VLMEVALKIT as the evaluation tools. Please refer to [QuickStart](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) for installation.

We evaluate the models with scripts in [evaluation.sh](scripts/evaluation.sh). You could modify the parameters for evaluating different benchmarks.

You should use the file [run.py](conv-llava/llava/eval/run.py) to replace with the original run file to evaluate the model.

```bash
eval_dataset="MMVet" # need openai api key
eval_dataset="MME MMBench_DEV_EN"

set the llava-path to the actual path of your convllava checkpoint
```

We would contribute the VLMEVALKIT to support our model soon. 

We have plans to support lmm-evals in the future.