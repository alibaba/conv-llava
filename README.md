# ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models

**ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models**
[paper]() [coda]() 

Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, Bo Zheng

## Release
- [04/17] Our code and paper are released. 


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) for the dataset and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-2 and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.


## Contents
- [Install](#install)
- [Model Zoo]()
- [Dataset]()
- [Train](#train)
- [Evaluation](#evaluation)


## TODO

- [ ] Add [LMMs-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) supports.
- [ ] Add [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) supports.
- [ ] Add [xtuner](https://github.com/InternLM/xtuner) supports.
- [ ] Release weights.
- [ ] Release inference code.

## Install

1. Clone this repository and navigate to ConvLLaVA folder
```bash
git clone https://github.com/alibaba/conv-llava
cd conv-llava
```

1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


## LLaVA Weights
Please check out our [Model Zoo]() for all public LLaVA checkpoints, and the instructions of how to use the weights.

## Train

We use the following hyperparameters for training ConvLLaVA.

| Hyperparameters | Stage 1 | Stage 2 | Stage 3 |
| --------------- | ------- | ------- | ------- |
| Learning Rate   | 3e-4    | 2e-5    | 2e-5    |
| Batch Size      | 256     | 256     | 128     |
| Epochs          | 1       | 1       | 1       |
| Warmup Ratio    | 0.03    | 0.03    | 0.03    |
| Weight Decay    | 0       | 0       | 0       |
| Optimizer       | AdamW   | AdamW   | AdamW   |

The training scripts are in the [scripts](conv-llava/scripts):

- Projector Initialzation: conv-llava/scripts/stage_1.sh
- Vision Language Pretraining: conv-llava/scripts/stage_2.sh
- Instruction Tuning: conv-llava/scripts/stage_3.sh

### Download Vicuna checkpoints (automatically)

Our base model Vicuna v1.5, which is an instruction-tuned chatbot, will be downloaded automatically when you run our provided training scripts. You could also download it from [vicuna](https://github.com/lm-sys/FastChat#vicuna-weights).

## Evaluation

We use VLMEVALKIT to evaluate our model now. See [Evaluation.md](conv-llava/docs/Evaluation.md) for more details.

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{ge2024convllava,
    title={ConvLLaVA: Hierarchical Backbones as Visual
Encoder for Large Multimodal Models},
    author={Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, Bo Zheng},
    month={April},
    year={2024}
}
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase LLaVA built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
