# Evaluation

## VLMEvalKit

We use VLMEVALKIT as the evaluation tools. Please refer to [QuickStart](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md) for installation.

We evaluate the models with scripts in [evaluation.sh](scripts/evaluation.sh). You could modify the parameters for evaluating different benchmarks.

You should use the file [run.py](conv-llava/llava/eval/run.py) to replace with the original run file to evaluate the model.

```bash
eval_dataset="MMVet" # need openai api key
eval_dataset="MME MMBench_DEV_EN"

# set the llava-path to the actual path of your convllava checkpoint
```

We would contribute the VLMEVALKIT to support our model soon. 

## lmms-eval

If you want to use lmms-eval to evaluate the model. You need to first install the package:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .
```

You should use the file [eval-lmms.sh](conv-llava/llava/eval/eval-lmms.sh) to evaluate the model. You could modify the parameters for evaluating different benchmarks.


## RefCOCO

If you are interested in RefCOCO, we provide the code in  [refcoco.sh](scripts/refcoco.sh). 