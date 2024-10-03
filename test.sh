OPT=$1
DEVICE=$2
PORT=$3
CUDA_VISIBLE_DEVICES=$DEVICE \
python -m torch.distributed.launch  \
--nproc_per_node=1 --master_port=$PORT ram/test.py \
-opt $OPT \
--launcher pytorch
