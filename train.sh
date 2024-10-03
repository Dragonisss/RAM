OPT=$1

python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 ram/train.py \
--auto_resume \
-opt $OPT --launcher pytorch
