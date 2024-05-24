# you can refer to this issue: https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/15

export HF_HOME=/path-to-save-dir

path=/path-to-your-model

accelerate launch --num_processes=8  -m lmms_eval \
    --model llava \
    --model_args pretrained="${path}"  \
    --tasks mme \
    --batch_size 1 \
    --log_samples --log_samples_suffix convllava \
    --output_path ./logs/ 