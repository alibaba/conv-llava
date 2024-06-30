CHECKPOINT=/path/to/convllava

torchrun \
--nnodes=1 \
--node_rank=0 \
--master_addr=127.0.0.1 \
--nproc_per_node=8 \
--master_port=25908 \
llava/eval/evaluate_grounding.py --checkpoint ${CHECKPOINT} --out-dir output


