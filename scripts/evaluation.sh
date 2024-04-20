export LMUData=path_to_your_saved_data
export OMP_NUM_THREADS=1

eval_dataset="MMBench_DEV_EN"

llava_path=path_to_your_weights

work_dir=path_to_your_work_dir
gpu=2

# if you want to use chatgpt to evaluate, you need to set OPENAI_API_KEY
# export OPENAI_API_KEY="sk-1234"

torchrun --nproc-per-node=${gpu} llava/eval/run.py \
    --data ${eval_dataset} \
    --model llava_v1.5_7b \
    --verbose \
    --work-dir ${work_dir}/vlmeval \
    --llava-path=${llava_path}


